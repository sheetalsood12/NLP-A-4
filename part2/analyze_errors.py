import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

from load_data import load_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_sql', required=True)
    parser.add_argument('--predicted_records', required=True)
    parser.add_argument('--data_folder', default='data')
    parser.add_argument('--ground_truth_records', default='records/ground_truth_dev.pkl')
    parser.add_argument('--output_dir', default='analysis')
    parser.add_argument('--labels', help='Reviewed CSV containing row_index and error_type columns')
    args = parser.parse_args()
    texts = load_lines(Path(args.data_folder) / 'dev.nl')
    gold = load_lines(Path(args.data_folder) / 'dev.sql')
    predicted = load_lines(args.predicted_sql)
    with open(args.ground_truth_records, 'rb') as f:
        gold_records, _ = pickle.load(f)
    with open(args.predicted_records, 'rb') as f:
        records, errors = pickle.load(f)
    if len({len(texts), len(gold), len(predicted), len(gold_records), len(records), len(errors)}) != 1:
        raise ValueError('Dev file lengths differ.')
    labels = {}
    if args.labels:
        with open(args.labels, newline='') as f:
            labels = {int(row['row_index']): row['error_type'] for row in csv.DictReader(f) if row['error_type'].strip()}
    rows = []
    for i, (text, reference, query, expected, actual, error) in enumerate(zip(texts, gold, predicted, gold_records, records, errors)):
        mismatch = set(expected) != set(actual)
        if mismatch or error:
            rows.append(dict(row_index=i, input=text, gold_sql=reference, predicted_sql=query,
                             sql_error=error, gold_record_count=len(expected), predicted_record_count=len(actual),
                             error_type=labels.get(i, 'SQL execution error' if error else '')))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = ['row_index', 'input', 'gold_sql', 'predicted_sql', 'sql_error', 'gold_record_count',
               'predicted_record_count', 'error_type']
    with open(out / 'dev_errors.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {'total_dev_queries': len(texts), 'errors_for_review': len(rows),
               'unclassified': sum(not row['error_type'] for row in rows),
               'category_counts': dict(Counter(row['error_type'] for row in rows if row['error_type']))}
    (out / 'error_counts.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
