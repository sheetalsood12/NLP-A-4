import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p1utils = load('p1_utils', ROOT / 'part1/utils.py')
sys.modules['utils'] = p1utils
p1 = load('bert_main', ROOT / 'part1/main.py')
p2utils = load('p2_utils', ROOT / 'part2/utils.py')
sys.modules['utils'] = p2utils
sys.path.insert(0, str(ROOT / 'part2'))
import load_data
import t5_utils
import train_t5


class FakeTokenizer:
    unk_token_id, eos_token_id = 2, 1

    def __call__(self, texts, **kwargs):
        return {'input_ids': [[3 + ord(c) % 20 for c in text] + [1] for text in texts]}


def test_typos_preserve_labels_negations_and_format():
    row = {'text': "I never dislike this movie. It isn't awful.<br /> " * 30, 'label': 1}
    original = row.copy()
    a, b = p1utils.custom_transform(row), p1utils.custom_transform(row)
    assert a == b and row == original
    assert a['label'] == 1 and a['text'] != row['text']
    assert a['text'].count('never') == 30 and a['text'].count("isn't") == 30
    assert a['text'].count('<br />') == 30
    assert len(a['text']) == len(row['text'])
    assert len(a['text'].split()) == len(row['text'].split())


def test_augmentation_adds_exactly_5000_training_examples(monkeypatch):
    train = Dataset.from_dict({'text': [f'Training review {i} great acting' for i in range(5001)],
                               'label': [i % 2 for i in range(5001)]})
    monkeypatch.setattr(p1, 'tokenizer', FakeTokenizer())
    loader = p1.create_augmented_dataloader(SimpleNamespace(batch_size=8), {'train': train})
    assert len(loader.dataset) == 10001
    labels = [int(value) for value in loader.dataset['labels']]
    assert labels[:5001] == train['label']
    assert labels[5001:] == train.shuffle(seed=42).select(range(5000))['label']


def test_bert_training_and_resume(tmp_path, monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
        def forward(self, input_ids, labels):
            return SimpleNamespace(loss=torch.nn.functional.cross_entropy(self.layer(input_ids), labels))
        def save_pretrained(self, directory, **kwargs):
            torch.save(self.state_dict(), Path(directory) / 'tiny.pt')
    monkeypatch.setattr(p1, 'tokenizer', None)
    torch.manual_seed(0)
    model = Tiny()
    x = torch.tensor([[1., 0.], [0., 1.]]).repeat(8, 1)
    y = torch.tensor([0, 1]).repeat(8)
    loader = torch.utils.data.DataLoader([dict(input_ids=a, labels=b) for a, b in zip(x, y)],batch_size=4)
    args = SimpleNamespace(learning_rate=.1, num_epochs=4, batch_size=4, resume=False)
    before = model(x, y).loss.item()
    p1.do_train(args, model, loader, tmp_path)
    assert model(x, y).loss.item() < before / 2
    trained = {k: v.clone() for k,v in model.state_dict().items()}
    args.resume = True
    p1.do_train(args, model, loader, tmp_path)
    assert all(torch.equal(v, trained[k]) for k,v in model.state_dict().items())
    assert len(json.loads((tmp_path / 'history.json').read_text())) == 4


def test_t5_alignment_and_test_order(tmp_path):
    (tmp_path / 'train.nl').write_text('a\nbc\n')
    (tmp_path / 'train.sql').write_text('SELECT 1\nSELECT 22\n')
    (tmp_path / 'test.nl').write_text('z\ny\n')
    tokenizer = FakeTokenizer()
    ds = load_data.T5Dataset(tmp_path,'train',tokenizer)
    enc, mask, dec, targets, initial = load_data.normal_collate_fn([ds[0],ds[1]])
    assert initial.tolist() == [[0],[0]] and torch.equal(dec[:,1:],targets[:,:-1])
    assert targets[0,-1].item() == 0 and targets[1,-1].item() == 1
    assert mask.equal(enc.ne(0).long())
    dl = load_data.get_dataloader(1,'test',tmp_path,tokenizer)
    assert list(dl.dataset.texts) == ['z','y']
    assert isinstance(dl.sampler,torch.utils.data.SequentialSampler)


def test_t5_loss_and_gradient_ignore_padding():
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_ff=32,num_layers=1,
        num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0.))
    batch = load_data.normal_collate_fn([(torch.tensor([3,1]),torch.tensor([4,1])),
                                       (torch.tensor([3,5,1]),torch.tensor([5,6,1]))])
    enc,mask,dec,targets,_ = batch
    logits=model(input_ids=enc,attention_mask=mask,decoder_input_ids=dec,
                 decoder_attention_mask=train_t5.decoder_mask(dec)).logits
    manual=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=0)
    labels=targets.masked_fill(targets.eq(0),-100)
    builtin=model(input_ids=enc,attention_mask=mask,labels=labels,
                  decoder_attention_mask=train_t5.decoder_mask(dec)).loss
    assert torch.allclose(manual,builtin,atol=1e-6)
    weights=model.shared.weight.detach().clone()
    args=SimpleNamespace()
    train_t5.train_epoch(args,model,[batch],torch.optim.AdamW(model.parameters(),lr=.01),None)
    assert not torch.equal(weights,model.shared.weight)


def test_checkpoint_roundtrip(tmp_path):
    args=SimpleNamespace(output_root=str(tmp_path),finetune=True,experiment_name='check')
    model=T5ForConditionalGeneration(T5Config(vocab_size=32,d_model=16,d_ff=32,num_layers=1,num_heads=2))
    t5_utils.save_model(t5_utils.checkpoint_directory(args),model,True)
    restored=t5_utils.load_model_from_checkpoint(args,True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())


def test_sql_records_order_errors_and_readonly(tmp_path,monkeypatch):
    db=tmp_path/'data.db'
    conn=sqlite3.connect(db);conn.execute('CREATE TABLE t (x INTEGER)')
    conn.execute('INSERT INTO t VALUES (7)');conn.commit();conn.close()
    monkeypatch.setattr(p2utils,'DB_PATH',str(db))
    records,errors=p2utils.compute_records(['SELECT x FROM t','SELECT 3','DELETE FROM t','SELECT x FROM t'])
    assert records==[[(7,)],[(3,)],[],[(7,)]]
    assert errors[2] and not errors[0]
    path=tmp_path/'output.sql';rec=tmp_path/'output.pkl'
    p2utils.save_queries_and_records(['SELECT x FROM t'],path,rec)
    metrics=p2utils.compute_metrics(str(path),str(path),str(rec),str(rec))
    assert metrics[0:2]==(1.,1.) and metrics[2]==pytest.approx(1.)
