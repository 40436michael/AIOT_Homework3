from pathlib import Path
import json
import argparse
from pprint import pformat

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import joblib

from src.spam_pipeline.preprocess import preprocess_text, normalize_and_mask

def _identity(x):
    # top-level identity function so it is picklable
    return x


def build_pipeline(ngram_range=(1, 1), max_features=5000, use_stop_words=True, C=1.0, min_df=1, sublinear_tf=False, class_weight=None):
    tfidf = TfidfVectorizer(
        preprocessor=_identity,
        tokenizer=str.split,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english' if use_stop_words else None,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    clf = LogisticRegression(solver='liblinear', C=C, max_iter=1000, class_weight=class_weight)
    pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])
    return pipe


def train(csv_path: Path, model_dir: Path, test_size=0.2, seed=42, ngram_range=(1,1), min_df=1, sublinear_tf=False, class_weight=None, C=1.0, max_features=5000, eval_threshold=0.5):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns and 'col_0' not in df.columns:
        raise ValueError('Input CSV must contain label column (label or col_0)')

    # support different label column names
    label_col = 'label' if 'label' in df.columns else 'col_0'
    text_col = 'message' if 'message' in df.columns else ('message_clean' if 'message_clean' in df.columns else None)
    if text_col is None:
        # try first non-label column
        cols = [c for c in df.columns if c != label_col]
        text_col = cols[0]

    # preprocess messages using normalize_and_mask
    df['message_clean'] = df[text_col].fillna('').apply(normalize_and_mask)

    X = df['message_clean'].values
    y = df[label_col].map(lambda v: 1 if str(v).strip().lower() == 'spam' else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    pipe = build_pipeline(ngram_range=ngram_range, max_features=max_features, use_stop_words=True, C=C, min_df=min_df, sublinear_tf=sublinear_tf, class_weight=class_weight)
    pipe.fit(X_train, y_train)

    y_score = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= eval_threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    roc = None
    try:
        roc = roc_auc_score(y_test, y_score)
    except Exception:
        roc = None

    # additional aggregated metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'logreg_pipeline.joblib'
    joblib.dump(pipe, model_path)

    metrics = {'classification_report': report, 'roc_auc': roc, 'precision': precision, 'recall': recall, 'f1': f1, 'eval_threshold': eval_threshold}
    (model_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # 為避免在日誌中洩漏絕對路徑或使用者名稱，只列印檔名與摘要資訊
    try:
        model_name = model_path.name
    except Exception:
        model_name = str(model_path)
    print('Trained model saved to (filename):', model_name)
    print('Metrics:')
    print(pformat(metrics))

    return model_path, metrics


def main():
    parser = argparse.ArgumentParser(description='Train spam classifier')
    parser.add_argument('--input', type=Path, default=Path('data/processed/sms_spam_clean.csv'))
    parser.add_argument('--model-dir', type=Path, default=Path('models'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_path, metrics = train(args.input, args.model_dir, test_size=args.test_size, seed=args.seed)

    return 0


if __name__ == '__main__':
    main()
