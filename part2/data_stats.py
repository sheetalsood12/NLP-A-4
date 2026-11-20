import argparse
import json
from pathlib import Path

from transformers import T5TokenizerFast
from load_data import MODEL_NAME, INPUT_PREFIX, load_tokenizer, load_lines


def summarize(inputs, outputs, tokenizer):
    x = tokenizer(inputs, add_special_tokens=True, truncation=False)['input_ids']
    y = tokenizer(outputs, add_special_tokens=True, truncation=False)['input_ids']
    special = set(tokenizer.all_special_ids)
    return {'examples': len(x), 'mean_sentence_length': sum(map(len, x)) / len(x),
            'mean_sql_length': sum(map(len, y)) / len(y),
            'nl_vocabulary_size': len({t for ids in x for t in ids if t not in special}),
            'sql_vocabulary_size': len({t for ids in y for t in ids if t not in special}),
            'max_sentence_length': max(map(len, x)), 'max_sql_length': max(map(len, y)),
            'sql_unknown_tokens': sum(ids.count(tokenizer.unk_token_id) for ids in y)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='data')
    parser.add_argument('--output', default='metrics/data_stats.json')
    args = parser.parse_args()
    original = T5TokenizerFast.from_pretrained(MODEL_NAME)
    processed = load_tokenizer()
    result = {'definition': 'Lengths include EOS, exclude padding; vocabulary counts unique observed non-special token IDs.',
              'before': {}, 'after': {}, 'model': MODEL_NAME,
              'processing': 'Strip outer whitespace, prefix inputs with translate English to SQL: , add ordinary < and <= tokens; no truncation.'}
    for split in ['train', 'dev']:
        x = load_lines(Path(args.data_folder) / (split + '.nl'))
        y = load_lines(Path(args.data_folder) / (split + '.sql'))
        result['before'][split] = summarize(x, y, original)
        result['after'][split] = summarize([INPUT_PREFIX + text for text in x], y, processed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
