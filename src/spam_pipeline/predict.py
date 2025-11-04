from pathlib import Path
import argparse
import csv
import joblib
import json

def load_pipeline(path: Path):
    return joblib.load(path)


def predict_single(pipe, text: str):
    prob = pipe.predict_proba([text])[0][1]
    label = 'spam' if prob >= 0.5 else 'ham'
    return {'text': text, 'prob_spam': float(prob), 'label': label}


def predict_csv(pipe, in_path: Path, out_path: Path, text_col: str = 'message'):
    import pandas as pd
    df = pd.read_csv(in_path)
    if text_col not in df.columns:
        raise ValueError(f'Text column {text_col} not found')
    texts = df[text_col].fillna('').astype(str).tolist()
    probs = pipe.predict_proba(texts)[:, 1]
    labels = ['spam' if p >= 0.5 else 'ham' for p in probs]
    df['prob_spam'] = probs
    df['pred_label'] = labels
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Predict with saved spam pipeline')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--input-csv', type=Path, help='CSV input for batch predict')
    parser.add_argument('--output-csv', type=Path, help='CSV output path for batch predict')
    parser.add_argument('--text-col', type=str, default='message')
    args = parser.parse_args()

    pipe = load_pipeline(args.model)

    if args.text:
        res = predict_single(pipe, args.text)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.input_csv:
        out = args.output_csv or Path('predictions.csv')
        res_path = predict_csv(pipe, args.input_csv, out, text_col=args.text_col)
        print('Wrote predictions to', res_path)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
