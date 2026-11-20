from pathlib import Path

import torch
from transformers import (T5ForConditionalGeneration, T5Config,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

MODEL_NAME = 'google-t5/t5-small'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_wandb(args):
    import wandb
    wandb.init(project='nlp-assignment4', name=args.experiment_name, config=vars(args))


def checkpoint_directory(args):
    model_type = 'ft' if args.finetune else 'scr'
    return Path(args.output_root) / 'checkpoints' / f'{model_type}_experiments' / args.experiment_name


def initialize_model(args):
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    else:
        model = T5ForConditionalGeneration(T5Config.from_pretrained(MODEL_NAME))
    if getattr(args, 'freeze_encoder', False):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    return model.to(DEVICE)


def mkdir(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def save_model(checkpoint_dir, model, best, tokenizer=None):
    destination = Path(checkpoint_dir) / ('best' if best else 'last')
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(destination)


def load_model_from_checkpoint(args, best):
    path = checkpoint_directory(args) / ('best' if best else 'last')
    return T5ForConditionalGeneration.from_pretrained(path).to(DEVICE)


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    return optimizer, initialize_scheduler(args, optimizer, epoch_length)


def initialize_optimizer(args, model):
    decay = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
    decay = {name for name in decay if 'bias' not in name}
    groups = [
        {'params': [p for n, p in model.named_parameters() if n in decay and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n not in decay and p.requires_grad],
         'weight_decay': 0.0},
    ]
    if args.optimizer_type != 'AdamW':
        raise ValueError('Only AdamW is supported.')
    return torch.optim.AdamW(groups, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))


def initialize_scheduler(args, optimizer, epoch_length):
    total = epoch_length * args.max_n_epochs
    warmup = epoch_length * args.num_warmup_epochs
    if args.scheduler_type == 'none':
        return None
    if args.scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, warmup, total)
    if args.scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(optimizer, warmup, total)
    raise ValueError(args.scheduler_type)


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        if not isinstance(child, tuple(forbidden_layer_types)):
            result.extend(f'{name}.{n}' for n in get_parameter_names(child, forbidden_layer_types))
    result.extend(model._parameters.keys())
    return result
