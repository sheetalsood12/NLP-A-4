import argparse
import math
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from utils import custom_transform, TYPO_PROBABILITY, TRANSFORM_SEED

MODEL_NAME = 'bert-base-cased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = None


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


def tokenized(split):
    result = split.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=False)
    result = result.rename_column('label', 'labels')
    result.set_format('torch')
    return result


def validation_score(model, loader, bf16=False):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model_device) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                outputs = model(**batch)
            size = len(batch['labels'])
            loss_sum += outputs.loss.item() * size
            correct += (outputs.logits.argmax(-1) == batch['labels']).sum().item()
            count += size
    return {'validation_loss': loss_sum / count, 'validation_accuracy': correct / count}


def do_train(args, model, train_dataloader, save_dir='./out', validation_loader=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_device = next(model.parameters()).device
    bf16 = args.bf16 and model_device.type == 'cuda'
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bfloat16.')
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * len(train_dataloader)
    scheduler = get_scheduler('linear', optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warmup_ratio * total_steps), num_training_steps=total_steps)
    settings = dict(epochs=args.num_epochs, lr=args.learning_rate, batch=args.batch_size,
        steps_per_epoch=len(train_dataloader), warmup=args.warmup_ratio,
        clip=args.max_grad_norm, validation_fraction=args.validation_fraction,
        seed=args.seed, bf16=bf16, augmented=args.train_augmented, debug=args.debug_train)
    history, start_epoch, best_accuracy, best_loss, best_epoch = [], 0, -1., float('inf'), 0
    last = save_dir / 'last'
    if args.resume:
        state = torch.load(last / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['settings'] != settings:
            raise ValueError('Resume requires identical training and split settings.')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, start_epoch = state['history'], state['epoch']
        best_accuracy, best_loss, best_epoch = state['best_accuracy'], state['best_loss'], state['best_epoch']
        # Restore training randomness
        random.setstate(state['random_state']); np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        if model_device.type == 'cuda': torch.cuda.set_rng_state_all(state['cuda_state'])
    (save_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2))
    print('TRAIN_CONFIG', settings, flush=True)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    for epoch in range(start_epoch, args.num_epochs):
        model.train()  # Validation switches to eval mode; restore dropout every epoch.
        total_loss, total_examples = 0., 0
        for step, batch in enumerate(train_dataloader):
            batch = {key: value.to(model_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                loss = model(**batch).loss
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            # Check encoder receives gradients
            if epoch == 0 and step == 0:
                encoder = next((p for n,p in model.named_parameters() if 'word_embeddings.weight' in n), None)
                if encoder is not None:
                    assert encoder.grad is not None and encoder.grad.norm().item() > 0, 'Encoder is not receiving gradients'
                print('GRADIENT_CHECK_OK; first loss:', loss.item(), 'unclipped norm:', float(grad_norm), flush=True)
            optimizer.step(); scheduler.step()
            count = len(batch['labels'])
            total_loss += loss.item() * count; total_examples += count
            if (step+1) % 100 == 0:
                print('PROGRESS', epoch+1, step+1, '/', len(train_dataloader),
                      'mean_loss', round(total_loss/total_examples,5), 'lr', scheduler.get_last_lr()[0], flush=True)
        metrics = validation_score(model, validation_loader, bf16) if validation_loader is not None else {}
        record = dict(epoch=epoch+1, train_loss=total_loss/total_examples, **metrics)
        history.append(record); print('EPOCH_RESULT', record, flush=True)
        score = metrics.get('validation_accuracy', 0.)
        val_loss = metrics.get('validation_loss', record['train_loss'])
        if validation_loader is None or score > best_accuracy or (score == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = score, val_loss, epoch+1
            model.save_pretrained(save_dir, safe_serialization=True)
            if tokenizer is not None: tokenizer.save_pretrained(save_dir)
        last.mkdir(exist_ok=True)
        model.save_pretrained(last, safe_serialization=True)
        state = dict(epoch=epoch+1, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            history=history, settings=settings, best_accuracy=best_accuracy, best_loss=best_loss,
            best_epoch=best_epoch, random_state=random.getstate(), numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all() if model_device.type=='cuda' else [])
        # Save before replacing checkpoint
        torch.save(state, last/'training_state.tmp')
        os.replace(last/'training_state.tmp', last/'training_state.pt')
        (save_dir/'history.json').write_text(json.dumps(history, indent=2))
        selection = dict(best_epoch=best_epoch, selection='fixed final training epoch')
        if validation_loader is not None:
            selection.update(validation_accuracy=best_accuracy, validation_loss=best_loss,
                             selection='training-split validation only')
        (save_dir/'selection.json').write_text(json.dumps(selection, indent=2))
    print('TRAINING_COMPLETE; selected epoch:', best_epoch, flush=True)


def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    model.eval()
    correct, count = 0, 0
    with open(out_file, 'w') as stream, torch.no_grad():
        for batch in tqdm(eval_dataloader, desc='Evaluation'):
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            count += len(predictions)
            for pred, label in zip(predictions.tolist(), batch['labels'].tolist()):
                stream.write(f'{pred}\n{label}\n')
    score = {'accuracy': correct / count, 'correct': correct, 'count': count}
    Path(str(out_file) + '.metrics.json').write_text(json.dumps(score, indent=2))
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return score


def create_augmented_dataloader(args, dataset):
    train = dataset['train']
    if len(train) < 5000:
        raise ValueError('Augmentation needs at least 5,000 training examples.')
    extra = train.shuffle(seed=42).select(range(5000)).map(custom_transform, load_from_cache_file=False)
    augmented = concatenate_datasets([train, extra])
    return DataLoader(tokenized(augmented), shuffle=True, batch_size=args.batch_size)


def create_transformed_dataloader(args, dataset, debug_transformation=False):
    split = dataset['test']
    if debug_transformation:
        examples = split.shuffle(seed=42).select(range(min(5, len(split))))
        for example in examples:
            print('Original:', example['text'])
            print('Transformed:', custom_transform(example)['text'])
            print('Label:', example['label'], '\n')
        return None
    transformed = split.map(custom_transform, load_from_cache_file=False)
    return DataLoader(tokenized(transformed), batch_size=args.batch_size)


def main():
    parser = argparse.ArgumentParser()
    for flag in ['train', 'train_augmented', 'eval', 'eval_transformed', 'debug_train',
                 'debug_transformation', 'resume']:
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--model_dir', default='./out')
    parser.add_argument('--output_root', default='.')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--validation_fraction', type=float, default=0.0,
                        help='Use a positive fraction for development; 0 trains on all examples.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 0.5: parser.error('Use a validation fraction in [0, 0.5)')
    if not 0 <= args.warmup_ratio < 1: parser.error('Invalid warmup ratio')
    if args.max_grad_norm <= 0: parser.error('Gradient norm must be positive')
    if args.train and args.train_augmented:
        parser.error('Run original and augmented training as separate commands.')
    if args.debug_train and args.train_augmented:
        parser.error('The 5,000-example augmentation uses the full training split.')
    if args.num_epochs < 1 or args.batch_size < 1:
        parser.error('num_epochs and batch_size must be positive.')
    if not any([args.train, args.train_augmented, args.eval, args.eval_transformed, args.debug_transformation]):
        parser.error('Choose a training or evaluation action.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = {}
    if args.train or args.train_augmented:
        dataset['train'] = load_dataset('imdb', split='train')
    if args.eval or args.eval_transformed or args.debug_transformation:
        dataset['test'] = load_dataset('imdb', split='test')
    if args.debug_transformation:
        create_transformed_dataloader(args, dataset, True)
        return
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.debug_train:
        if 'train' in dataset:
            dataset['train'] = dataset['train'].shuffle(seed=42).select(range(4000))
        if 'test' in dataset:
            dataset['test'] = dataset['test'].shuffle(seed=42).select(range(1000))
    if args.train or args.train_augmented:
        name = 'out_debug' if args.debug_train else ('out_augmented' if args.train_augmented else 'out')
        save_dir = root / name
        if not args.resume and (save_dir / 'config.json').exists():
            raise FileExistsError(f'{save_dir} already has a model; use --resume or a new --output_root.')
        # Split only the training set. Augmentation is applied after the split.
        original_train = dataset['train']
        labels = np.array(original_train['label'])
        assert set(labels.tolist()) == {0, 1}, 'Expected binary IMDB labels'
        rng = np.random.default_rng(args.seed)
        train_indices, val_indices = [], []
        for label in (0, 1):
            indices = rng.permutation(np.flatnonzero(labels == label))
            n_val = max(1, int(len(indices)*args.validation_fraction)) if args.validation_fraction else 0
            val_indices.extend(indices[:n_val].tolist())
            train_indices.extend(indices[n_val:].tolist())
        assert not set(train_indices).intersection(val_indices)
        dataset['train'] = original_train.select(sorted(train_indices))
        validation_loader = (DataLoader(tokenized(original_train.select(sorted(val_indices))),
                                       batch_size=args.batch_size) if val_indices else None)
        save_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = dict(seed=args.seed, train_indices=sorted(train_indices), validation_indices=sorted(val_indices),
            source='imdb/train', debug=args.debug_train)
        (save_dir/'split.json').write_text(json.dumps(split_manifest))
        print('DATA_AUDIT', 'train', len(train_indices), 'validation', len(val_indices),
              'labels', dict(zip(*np.unique(labels, return_counts=True))), flush=True)
        loader = (create_augmented_dataloader(args, dataset) if args.train_augmented else
                  DataLoader(tokenized(dataset['train']), shuffle=True, batch_size=args.batch_size))
        model_source = str(save_dir/'last') if args.resume else MODEL_NAME
        model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2).to(device)
        do_train(args, model, loader, save_dir, validation_loader)
        args.model_dir = str(save_dir)
        del model, loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    name = Path(args.model_dir).name
    if args.eval:
        loader = DataLoader(tokenized(dataset['test']), batch_size=args.batch_size)
        print('Score:', do_eval(loader, args.model_dir, root / f'{name}_original.txt'))
    if args.eval_transformed:
        loader = create_transformed_dataloader(args, dataset)
        print('Score:', do_eval(loader, args.model_dir, root / f'{name}_transformed.txt'))
        (root / 'transformation_config.json').write_text(json.dumps(
            {'probability': TYPO_PROBABILITY, 'seed': TRANSFORM_SEED, 'type': 'internal QWERTY substitution'}, indent=2))


if __name__ == '__main__':
    main()
