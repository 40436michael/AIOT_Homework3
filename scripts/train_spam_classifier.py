#!/usr/bin/env python
"""CLI wrapper to train spam classifier with tunable options"""
import argparse
from pathlib import Path

from src.spam_pipeline.train import train


def parse_ngram(s: str):
    parts = s.split(',')
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description='Train spam classifier (wrapper)')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngram-range', type=str, default='1,1')
    parser.add_argument('--min-df', type=int, default=1)
    parser.add_argument('--sublinear-tf', action='store_true')
    parser.add_argument('--class-weight', type=str, choices=['none','balanced'], default='none')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--eval-threshold', type=float, default=0.5)
    args = parser.parse_args()

    class_weight = None if args.class_weight == 'none' else 'balanced'
    ngram = parse_ngram(args.ngram_range)

    train(args.input, args.model_dir, test_size=args.test_size, seed=args.seed, ngram_range=ngram, min_df=args.min_df, sublinear_tf=args.sublinear_tf, class_weight=class_weight, C=args.C, max_features=args.max_features, eval_threshold=args.eval_threshold)


if __name__ == '__main__':
    main()
