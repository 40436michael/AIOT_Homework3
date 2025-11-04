#!/usr/bin/env python
"""Simple visualization CLI: class distribution, confusion matrix, ROC/PR curves"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from src.spam_pipeline.preprocess import normalize_and_mask


def plot_class_distribution(df, label_col='label', out_dir=Path('reports/visualizations')):
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = df[label_col].value_counts()
    fig = counts.plot(kind='bar', title='Class distribution').get_figure()
    fig.savefig(out_dir / 'class_distribution.png')
    plt.close(fig)


def plot_confusion_roc_pr(model_path: Path, df: pd.DataFrame, label_col='label', text_col='message_clean', out_dir=Path('reports/visualizations')):
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(model_path)
    texts = df[text_col].fillna('').apply(normalize_and_mask).tolist()
    y_true = df[label_col].map(lambda v: 1 if str(v).strip().lower() == 'spam' else 0).values
    probs = pipe.predict_proba(texts)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_true, preds)
    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    fig.savefig(out_dir / 'confusion_matrix.png')
    plt.close(fig)

    # ROC & PR
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[1].plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_dir / 'roc_pr.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize spam dataset and model')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--label-col', type=str, default='label')
    parser.add_argument('--text-col', type=str, default='message_clean')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_class_distribution(df, label_col=args.label_col)
    model_path = args.models_dir / 'logreg_pipeline.joblib'
    if model_path.exists():
        plot_confusion_roc_pr(model_path, df, label_col=args.label_col, text_col=args.text_col)
        print('Saved visualizations to reports/visualizations')
    else:
        print('Model not found at', model_path, '- only class distribution saved')


if __name__ == '__main__':
    main()
