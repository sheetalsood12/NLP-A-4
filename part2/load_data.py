from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0
MODEL_NAME = 'google-t5/t5-small'
INPUT_PREFIX = 'translate English to SQL: '
SQL_TOKENS = ['<', '<=']


def load_tokenizer(source=MODEL_NAME):
    tokenizer = T5TokenizerFast.from_pretrained(source)
    tokenizer.add_tokens(SQL_TOKENS)
    return tokenizer


class T5Dataset(Dataset):
    def __init__(self, data_folder, split, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
        self.split = split
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        self.texts = load_lines(Path(data_folder) / f'{split}.nl')
        self.queries = None if split == 'test' else load_lines(Path(data_folder) / f'{split}.sql')
        if self.queries is not None and len(self.texts) != len(self.queries):
            raise ValueError(f'{split}: input and SQL line counts differ.')
        self.encoder_ids = tokenizer([INPUT_PREFIX + text for text in self.texts],
                                     add_special_tokens=True, truncation=False)['input_ids']
        self.target_ids = None
        if self.queries is not None:
            self.target_ids = tokenizer(self.queries, add_special_tokens=True, truncation=False)['input_ids']
            for i, ids in enumerate(self.target_ids):
                if tokenizer.unk_token_id in ids:
                    raise ValueError(f'{split} SQL row {i} contains an unknown token; check tokenization.')
                if ids[-1] != tokenizer.eos_token_id:
                    raise ValueError('SQL targets must end with EOS.')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoder = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        if self.target_ids is None:
            return encoder
        return encoder, torch.tensor(self.target_ids[idx], dtype=torch.long)


def normal_collate_fn(batch):
    encoders, targets = zip(*batch)
    encoder_ids = pad_sequence(encoders, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    decoder_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    encoder_ids = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, encoder_ids.ne(PAD_IDX).long(), torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)


def get_dataloader(batch_size, split, data_folder='data', tokenizer=None):
    dataset = T5Dataset(data_folder, split, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                      collate_fn=test_collate_fn if split == 'test' else normal_collate_fn)


def load_t5_data(batch_size, test_batch_size, data_folder='data', tokenizer=None):
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer()
    return (get_dataloader(batch_size, 'train', data_folder, tokenizer),
            get_dataloader(test_batch_size, 'dev', data_folder, tokenizer),
            get_dataloader(test_batch_size, 'test', data_folder, tokenizer))


def load_lines(path):
    with open(path, encoding='utf-8') as stream:
        return [line.strip() for line in stream]


def load_prompting_data(data_folder):
    root = Path(data_folder)
    return tuple(load_lines(root / name) for name in ['train.nl', 'train.sql', 'dev.nl', 'dev.sql', 'test.nl'])
