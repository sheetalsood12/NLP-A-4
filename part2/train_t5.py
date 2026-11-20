import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from t5_utils import (initialize_model, initialize_optimizer_and_scheduler, save_model,
                      load_model_from_checkpoint, setup_wandb, checkpoint_directory)
from load_data import load_t5_data, load_tokenizer
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0


def get_args():
    parser = argparse.ArgumentParser(description='T5 text to SQL training')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--optimizer_type', choices=['AdamW'], default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_type', choices=['none', 'linear', 'cosine'], default='linear')
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=20)
    parser.add_argument('--patience_epochs', type=int, default=5)
    parser.add_argument('--experiment_name', default='experiment')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_root', default='.')
    parser.add_argument('--data_folder', default='data')
    parser.add_argument('--ground_truth_records', default='records/ground_truth_dev.pkl')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--predict_test', action='store_true')
    args = parser.parse_args()
    for field in ['batch_size', 'test_batch_size', 'max_n_epochs', 'patience_epochs', 'num_beams', 'max_new_tokens']:
        if getattr(args, field) < 1:
            parser.error(f'{field} must be positive.')
    if args.num_warmup_epochs >= args.max_n_epochs:
        parser.error('Warmup must be shorter than training.')
    return args


def output_paths(args, split):
    root = Path(args.output_root)
    model_type = 'ft' if args.finetune else 'scr'
    stem = f't5_{model_type}_{args.experiment_name}_{split}'
    for directory in ['results', 'records', 'metrics']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root / 'results' / (stem + '.sql'), root / 'records' / (stem + '.pkl')


def decoder_mask(decoder_input):
    mask = decoder_input.ne(PAD_IDX).long()
    # Keep decoder start visible
    mask[:, 0] = 1
    return mask


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss, total_tokens = 0., 0
    for encoder, mask, decoder, targets, _ in tqdm(train_loader, desc='Training'):
        encoder, mask, decoder, targets = [x.to(DEVICE) for x in (encoder, mask, decoder, targets)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=encoder, attention_mask=mask, decoder_input_ids=decoder,
                       decoder_attention_mask=decoder_mask(decoder), use_cache=False).logits
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_IDX)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        count = targets.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * count
        total_tokens += count
    return total_loss / total_tokens


def generated_queries(args, model, encoder, mask, initial, tokenizer):
    tokens = model.generate(input_ids=encoder, attention_mask=mask, decoder_input_ids=initial,
                            max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                            do_sample=False)
    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # Keep one query per line without removing whitespace inside SQL string literals.
    return [text.replace('\n', ' ').replace('\r', ' ').strip() for text in texts]


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss, total_tokens, predictions = 0., 0, []
    tokenizer = dev_loader.dataset.tokenizer
    with torch.no_grad():
        for encoder, mask, decoder, targets, initial in tqdm(dev_loader, desc='Development'):
            encoder, mask, decoder, targets, initial = [x.to(DEVICE) for x in (encoder, mask, decoder, targets, initial)]
            logits = model(input_ids=encoder, attention_mask=mask, decoder_input_ids=decoder,
                           decoder_attention_mask=decoder_mask(decoder)).logits
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_IDX, reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.ne(PAD_IDX).sum().item()
            predictions.extend(generated_queries(args, model, encoder, mask, initial, tokenizer))
    assert len(predictions) == len(dev_loader.dataset)
    save_queries_and_records(predictions, str(model_sql_path), str(model_record_path))
    sql_em, record_em, record_f1, errors = compute_metrics(str(gt_sql_pth), str(model_sql_path),
                                                         str(gt_record_path), str(model_record_path))
    return total_loss / total_tokens, float(record_f1), float(record_em), float(sql_em), sum(bool(e) for e in errors) / len(errors)


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    predictions = []
    with torch.no_grad():
        for encoder, mask, initial in tqdm(test_loader, desc='Test inference'):
            predictions.extend(generated_queries(args, model, encoder.to(DEVICE), mask.to(DEVICE),
                                                 initial.to(DEVICE), test_loader.dataset.tokenizer))
    assert len(predictions) == len(test_loader.dataset)
    save_queries_and_records(predictions, str(model_sql_path), str(model_record_path))


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    directory = checkpoint_directory(args)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'training_state.pt'
    start_epoch, best_f1, waiting, history = 0, -1., 0, []
    if args.resume:
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        relevant = ['learning_rate', 'weight_decay', 'scheduler_type', 'num_warmup_epochs', 'max_n_epochs',
                    'batch_size', 'finetune', 'freeze_encoder', 'num_beams', 'max_new_tokens', 'seed']
        if any(state['args'][k] != getattr(args, k) for k in relevant):
            raise ValueError('Resume settings differ from the saved training run.')
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        start_epoch, best_f1, waiting, history = state['epoch'], state['best_f1'], state['waiting'], state['history']
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        if DEVICE.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_state'])
    if waiting >= args.patience_epochs:
        print('This run already reached its early stopping criterion.')
        return
    sql_path, record_path = output_paths(args, 'dev')
    for epoch in range(start_epoch, args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        sql_path, record_path = output_paths(args, f'dev_epoch{epoch + 1:02d}')
        loss, f1, em, query_em, errors = eval_epoch(args, model, dev_loader,
            Path(args.data_folder) / 'dev.sql', sql_path, args.ground_truth_records, record_path)
        row = dict(epoch=epoch + 1, train_loss=tr_loss, dev_loss=loss, record_f1=f1,
                   record_em=em, sql_em=query_em, error_rate=errors)
        history.append(row)
        print(json.dumps(row), flush=True)
        if args.use_wandb:
            import wandb
            wandb.log(row, step=epoch + 1)
        improved = f1 > best_f1
        waiting = 0 if improved else waiting + 1
        if improved:
            best_f1 = f1
            save_model(directory, model, best=True, tokenizer=train_loader.dataset.tokenizer)
        save_model(directory, model, best=False, tokenizer=train_loader.dataset.tokenizer)
        state = dict(epoch=epoch + 1, best_f1=best_f1, waiting=waiting, history=history, args=vars(args),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                     random_state=random.getstate(), numpy_state=np.random.get_state(),
                     torch_state=torch.get_rng_state(),
                     cuda_state=torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else [])
        torch.save(state, directory / 'training_state.tmp')
        os.replace(directory / 'training_state.tmp', state_path)
        (directory / 'history.json').write_text(json.dumps(history, indent=2))
        (directory / 'run_config.json').write_text(json.dumps(vars(args), indent=2))
        if waiting >= args.patience_epochs:
            break


def main():
    args = get_args()
    set_random_seeds(args.seed)
    directory = checkpoint_directory(args)
    if args.use_wandb:
        setup_wandb(args)
    if args.eval_only or args.resume:
        tokenizer = load_tokenizer(directory / ('best' if args.eval_only else 'last'))
        model = load_model_from_checkpoint(args, best=args.eval_only)
        if args.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
    else:
        if (directory / 'last' / 'config.json').exists():
            raise FileExistsError('Existing experiment: use --resume or a different --experiment_name.')
        tokenizer = load_tokenizer()
        model = initialize_model(args)
        model.resize_token_embeddings(len(tokenizer))
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size,
                                                       args.data_folder, tokenizer)
    if not args.eval_only:
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        del optimizer, scheduler, model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        model = load_model_from_checkpoint(args, best=True)
    sql_path, record_path = output_paths(args, 'dev')
    values = eval_epoch(args, model, dev_loader, Path(args.data_folder) / 'dev.sql',
                        sql_path, args.ground_truth_records, record_path)
    metrics = dict(zip(['loss', 'record_f1', 'record_em', 'sql_em', 'error_rate'], values))
    metrics['num_beams'] = args.num_beams
    metrics['max_new_tokens'] = args.max_new_tokens
    path = Path(args.output_root) / 'metrics' / f'{directory.parent.name}_{args.experiment_name}_dev_beam{args.num_beams}.json'
    path.write_text(json.dumps(metrics, indent=2))
    print('Best model dev metrics:', metrics)
    if args.predict_test:
        sql_path, record_path = output_paths(args, 'test')
        test_inference(args, model, test_loader, sql_path, record_path)
        print('Saved test outputs:', sql_path, record_path)


if __name__ == '__main__':
    main()
