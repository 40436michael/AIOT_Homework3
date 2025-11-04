#!/usr/bin/env python
"""CLI wrapper to run predictions using trained pipeline"""
import argparse
from pathlib import Path

from src.spam_pipeline.predict import load_pipeline, predict_single, predict_csv


def main():
    parser = argparse.ArgumentParser(description='Predict spam using saved pipeline')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--text', type=str)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--text-col', type=str, default='message')
    args = parser.parse_args()

    pipe = load_pipeline(args.model)
    if args.text:
        res = predict_single(pipe, args.text)
        print(res)
        return

    if args.input:
        out = args.output or Path('predictions.csv')
        predict_csv(pipe, args.input, out, text_col=args.text_col)
        print('Wrote predictions to', out)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
