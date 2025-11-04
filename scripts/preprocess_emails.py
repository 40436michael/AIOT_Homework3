#!/usr/bin/env python
"""CLI wrapper: preprocess emails CSV into cleaned dataset
Creates an output CSV with a `message_clean` (or configurable) column.
"""
import argparse
from pathlib import Path
import pandas as pd

from src.spam_pipeline.preprocess import normalize_and_mask


def main():
    parser = argparse.ArgumentParser(description='Preprocess SMS spam CSV')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-header', action='store_true', help='Input CSV has no header (label,col)')
    parser.add_argument('--label-col-index', type=int, default=0)
    parser.add_argument('--text-col-index', type=int, default=1)
    parser.add_argument('--output-text-col', type=str, default='message_clean')
    args = parser.parse_args()

    if args.no_header:
        df = pd.read_csv(args.input, header=None)
        df.columns = [f'col_{i}' for i in range(df.shape[1])]
    else:
        df = pd.read_csv(args.input)

    label_col = df.columns[args.label_col_index]
    text_col = df.columns[args.text_col_index]

    df[args.output_text_col] = df[text_col].fillna('').apply(normalize_and_mask)

    # ensure label column name is 'label' for downstream consistency
    if label_col != 'label':
        df = df.rename(columns={label_col: 'label'})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print('Wrote cleaned CSV to', args.output)


if __name__ == '__main__':
    main()
