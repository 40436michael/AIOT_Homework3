import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from collections import Counter
from typing import List, Tuple

st.set_page_config(page_title="Spam Classifier — Demo + Visualizations", layout="wide")

try:
    import seaborn as sns
except Exception:
    sns = None
    # graceful fallback: provide minimal replacements so app can run without seaborn
    class _FakeSNS:
        def barplot(self, x=None, y=None, ax=None, **kwargs):
            if ax is None:
                fig, ax = plt.subplots()
            ax.barh(y, x, color=kwargs.get('color', None))
            return ax

        def histplot(self, data, color=None, label=None, stat=None, kde=False, ax=None, fill=False, **kwargs):
            if ax is None:
                fig, ax = plt.subplots()
            density = (stat == 'density')
            ax.hist(data, bins=50, density=density, alpha=0.6, color=color, label=label)
            return ax

        def boxplot(self, x=None, y=None, data=None, ax=None, **kwargs):
            if ax is None:
                fig, ax = plt.subplots()
            if data is not None and y in data.columns:
                data.boxplot(column=y, by=x, ax=ax)
            return ax

        def kdeplot(self, data, label=None, fill=False, ax=None, **kwargs):
            if ax is None:
                fig, ax = plt.subplots()
            try:
                from scipy.stats import gaussian_kde
                import numpy as _np
                kde = gaussian_kde(data)
                xs = _np.linspace(min(data), max(data), 200)
                ax.plot(xs, kde(xs), label=label)
            except Exception:
                ax.hist(data, bins=50, density=True, alpha=0.4, label=label)
            return ax

    sns = _FakeSNS()

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
    classification_report,
)
from sklearn.calibration import calibration_curve

from src.spam_pipeline.preprocess import normalize_and_mask

MODEL_DEFAULT = Path("models/logreg_pipeline.joblib")


@st.cache_resource
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def load_artifacts(models_dir: str):
    """Load separate vectorizer + clf artifacts if present.

    Falls back gracefully if files missing.
    """
    vec_p = os.path.join(models_dir, "spam_tfidf_vectorizer.joblib")
    clf_p = os.path.join(models_dir, "spam_logreg_model.joblib")
    vec = clf = None
    if os.path.exists(vec_p):
        try:
            vec = joblib.load(vec_p)
        except Exception:
            vec = None
    if os.path.exists(clf_p):
        try:
            clf = joblib.load(clf_p)
        except Exception:
            clf = None
    # optional label mapping
    pos, neg = "spam", "ham"
    meta_p = os.path.join(models_dir, "spam_label_mapping.json")
    if os.path.exists(meta_p):
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
                pos = meta.get("positive", pos)
                neg = meta.get("negative", neg)
        except Exception:
            pass
    return vec, clf, pos, neg


def plot_confusion_matrix_fig(y_true, y_pred, labels=("ham", "spam")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_roc_pr_fig(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_calibration_fig(y_true, y_score, n_bins=10):
    fraction_of_pos, mean_pred_value = calibration_curve(y_true, y_score, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred_value, fraction_of_pos, "s-", label="Calibration")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_score_dist_fig(y_true, y_score):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(np.array(y_score)[np.array(y_true) == 0], color="C0", label="ham", stat="density", kde=True, ax=ax)
    sns.histplot(np.array(y_score)[np.array(y_true) == 1], color="C1", label="spam", stat="density", kde=True, ax=ax)
    ax.set_xlabel("Predicted spam probability")
    ax.set_title("Score distribution by true label")
    ax.legend()
    plt.tight_layout()
    return fig


def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def list_datasets() -> List[str]:
    paths: List[str] = []
    for root in ("datasets", os.path.join("datasets", "processed")):
        if os.path.isdir(root):
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if name.lower().endswith(".csv") and os.path.isfile(p):
                    paths.append(p)
    return sorted(paths)


def infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    label_candidates = [c for c in cols if c.lower() in ("label", "target", "col_0")]
    text_candidates = [c for c in cols if c.lower() in ("text", "message", "text_clean", "col_1")]
    label = label_candidates[0] if label_candidates else cols[0]
    text = text_candidates[0] if text_candidates else cols[-1]
    return label, text


def token_topn(series: pd.Series, topn: int) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)


def label_to_int(series: pd.Series, pos_label: str = "spam") -> np.ndarray:
    s = series.astype(str).str.lower()
    return (s == pos_label.lower()).astype(int).values


def demo_page():
    st.sidebar.title("SMS Spam Classifier — Demo")
    model_path = st.sidebar.text_input("Model path", str(MODEL_DEFAULT))
    st.sidebar.markdown("Upload a CSV with `message` column or type a message below.")
    threshold = st.sidebar.slider("Decision threshold for spam", 0.0, 1.0, 0.5, 0.01)

    pipe = load_model(model_path)
    if pipe is None:
        st.warning(f"Model not found at {model_path}. Please train and place model at this path.")

    st.title("SMS Spam Classification Demo")

    # Example buttons
    col1, col2 = st.columns(2)
    if col1.button("Use spam example"):
        st.session_state["example_text"] = "Free entry in 2 a wkly comp to win cash now! Call +44 906-170-1461"
    if col2.button("Use ham example"):
        st.session_state["example_text"] = "Hey, are we still meeting for coffee tonight?"

    normalize_batch = st.checkbox("Normalize inputs before prediction (recommended)", value=True)
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded is not None and pipe is not None:
        df = pd.read_csv(uploaded)
        text_col = st.text_input("Text column name", "message")
        if text_col not in df.columns:
            st.error(f"Column {text_col} not found in uploaded CSV")
        else:
            texts_raw = df[text_col].fillna("").astype(str).tolist()
            if normalize_batch:
                texts = [normalize_and_mask(t) for t in texts_raw]
            else:
                texts = texts_raw
            probs = pipe.predict_proba(texts)[:, 1]
            preds = (probs >= threshold).astype(int)
            labels = ["spam" if p else "ham" for p in preds]
            df["prob_spam"] = probs
            df["pred_label"] = labels
            st.write(df.head(50))
            st.download_button("Download predictions CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv")

            # If uploaded CSV contains ground truth label, show evaluation
            label_col = st.text_input("Ground-truth label column name (optional)", "label")
            if label_col in df.columns:
                y_true = df[label_col].map(lambda v: 1 if str(v).strip().lower() == "spam" else 0).values
                # Confusion matrix
                fig_cm = plot_confusion_matrix_fig(y_true, preds)
                st.pyplot(fig_cm)
                # ROC + PR side-by-side
                fig_rocpr = plot_roc_pr_fig(y_true, probs)
                st.pyplot(fig_rocpr)
                # Calibration curve
                try:
                    fig_cal = plot_calibration_fig(y_true, probs)
                    st.pyplot(fig_cal)
                except Exception:
                    st.write("Calibration plot failed to compute.")
                # Score distribution
                try:
                    fig_dist = plot_score_dist_fig(y_true, probs)
                    st.pyplot(fig_dist)
                except Exception:
                    st.write("Score distribution plot failed.")

                # Summary metrics and download
                try:
                    report = classification_report(y_true, preds, output_dict=True)
                except Exception:
                    report = None
                try:
                    roc_auc = auc(*roc_curve(y_true, probs)[:2])
                except Exception:
                    roc_auc = None
                try:
                    ap = average_precision_score(y_true, probs)
                except Exception:
                    ap = None
                metrics = {"classification_report": report, "roc_auc": roc_auc, "average_precision": ap, "threshold": float(threshold), "n_samples": int(len(df)), "timestamp": ts()}
                st.write("Metrics (summary):")
                # Display numeric metrics prominently
                if roc_auc is not None:
                    st.metric("ROC AUC", f"{roc_auc:.3f}")
                if ap is not None:
                    st.metric("Average Precision (AP)", f"{ap:.3f}")
                st.json({"roc_auc": roc_auc, "average_precision": ap})
                # include timestamp in suggested filename
                fname = f"metrics_{ts()}.json"
                st.download_button("Download metrics (JSON)", data=json.dumps(metrics).encode("utf-8"), file_name=fname)

    st.markdown("---")
    st.subheader("Single message prediction")
    # default example text support
    default_text = st.session_state.get("example_text", "Free entry in 2 a wkly comp to win cash now!")
    text = st.text_area("Enter message text", default_text)
    if st.button("Predict") and pipe is not None:
        cleaned = normalize_and_mask(text)
        prob = float(pipe.predict_proba([cleaned])[0][1])
        label = "spam" if prob >= threshold else "ham"
        st.metric("Predicted label", label)
        st.metric("Spam probability", f"{prob:.3f}")
        st.write("---")
        st.write("Input (raw):")
        st.write(text)
        st.write("Input (cleaned):")
        st.write(cleaned)


def viz_page():
    st.title("Spam/Ham Classifier — Visualizations")
    st.caption("Interactive dashboard for data distribution, token patterns, and model performance")

    with st.sidebar:
        st.header("Inputs")
        datasets = list_datasets()
        ds_index = 0
        if "datasets/processed/sms_spam_clean.csv" in datasets:
            ds_index = datasets.index("datasets/processed/sms_spam_clean.csv")
        ds_path = st.selectbox("Dataset CSV", datasets, index=ds_index)
        df = load_csv(ds_path)
        label_col, text_col = infer_cols(df)
        label_col = st.selectbox("Label column", options=list(df.columns), index=list(df.columns).index(label_col))
        text_col = st.selectbox("Text column", options=list(df.columns), index=list(df.columns).index(text_col))

        models_dir = st.text_input("Models dir", value="models")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        threshold = st.slider("Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

    st.subheader("Data Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Class distribution")
        counts = df[label_col].value_counts().sort_index()
        st.bar_chart(counts)
    with c2:
        st.write("Token replacements in cleaned text (approximate)")
        sample = df[text_col].astype(str)
        repl = {
            "<URL>": sample.str.count(r"<URL>").sum(),
            "<EMAIL>": sample.str.count(r"<EMAIL>").sum(),
            "<PHONE>": sample.str.count(r"<PHONE>").sum(),
            "<NUM>": sample.str.count(r"<NUM>").sum(),
        }
        st.table(pd.DataFrame.from_dict(repl, orient="index", columns=["count"]))

    st.subheader("Top Tokens by Class")
    topn = st.slider("Top-N tokens", min_value=10, max_value=40, value=20, step=5)
    col_a, col_b = st.columns(2)
    for label, col in [(counts.index[0], col_a), (counts.index[-1], col_b)]:
        with col:
            st.write(f"Class: {label}")
            top = token_topn(df.loc[df[label_col] == label, text_col], topn)
            if top:
                toks, freqs = zip(*top)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=list(freqs), y=list(toks), ax=ax, palette="viridis")
                ax.set_xlabel("frequency")
                ax.set_ylabel("token")
                st.pyplot(fig)
            else:
                st.info("No tokens found.")

    # Model-based visuals
    st.subheader("Model Performance (Test)")
    # try loading separate artifacts first, else try a single pipeline artifact
    vec, clf, pos_label, neg_label = load_artifacts(models_dir)
    pipeline_p = os.path.join(models_dir, "logreg_pipeline.joblib")
    if (vec is None or clf is None) and os.path.exists(pipeline_p):
        try:
            pipe_tmp = joblib.load(pipeline_p)
            if hasattr(pipe_tmp, "named_steps"):
                vec = vec or pipe_tmp.named_steps.get("tfidf")
                clf = clf or pipe_tmp.named_steps.get("clf")
        except Exception:
            pass

    if vec is not None and clf is not None:
        X = df[text_col].astype(str).fillna("")
        y = label_to_int(df[label_col], pos_label=pos_label)
        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        Xte_vec = vec.transform(Xte)
        proba = clf.predict_proba(Xte_vec)[:, 1]
        pred = (proba >= threshold).astype(int)

        # Confusion matrix
        cm = confusion_matrix(yte, pred)
        cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]) 
        st.write("Confusion matrix")
        st.dataframe(cm_df)

        # ROC/PR curves
        fpr, tpr, _ = roc_curve(yte, proba)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(yte, proba)
        pr_fig, pr_ax = plt.subplots(1, 2, figsize=(10, 4))
        pr_ax[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        pr_ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        pr_ax[0].set_title("ROC")
        pr_ax[0].set_xlabel("FPR")
        pr_ax[0].set_ylabel("TPR")
        PrecisionRecallDisplay = None
        try:
            from sklearn.metrics import PrecisionRecallDisplay
            PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=pr_ax[1])
        except Exception:
            pr_ax[1].plot(rec, prec)
        pr_ax[1].set_title("Precision-Recall")
        # display ROC AUC and AP as numeric metrics
        try:
            ap_val = average_precision_score(yte, proba)
        except Exception:
            ap_val = None
        if roc_auc is not None:
            st.metric("ROC AUC", f"{roc_auc:.3f}")
        if ap_val is not None:
            st.metric("Average Precision (AP)", f"{ap_val:.3f}")
        st.pyplot(pr_fig)

        # Threshold sweep small table
        st.write("Threshold sweep (precision/recall/f1)")
        ths = np.round(np.linspace(0.3, 0.8, 11), 3)
        rows = []
        for t in ths:
            p = (proba >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score

            rows.append({
                "threshold": t,
                "precision": float(precision_score(yte, p, zero_division=0)),
                "recall": float(recall_score(yte, p, zero_division=0)),
                "f1": float(f1_score(yte, p, zero_division=0)),
            })
        st.dataframe(pd.DataFrame(rows))

        # Live Inference
        st.subheader("Live Inference")
        # Provide two quick examples to try
        ex_spam = "Free entry in 2 a wkly comp to win cash now! Call +44 906-170-1461 to claim prize"
        ex_ham = "Ok, I'll see you at 7 pm for dinner. Thanks!"
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1:
            if st.button("Use spam example"):
                st.session_state["input_text"] = ex_spam
        with c_ex2:
            if st.button("Use ham example"):
                st.session_state["input_text"] = ex_ham

        # Text area bound to session_state so examples populate it
        if "input_text" not in st.session_state:
            st.session_state["input_text"] = ""
        user_text = st.text_area("Enter a message to classify", key="input_text")

        if st.button("Predict"):
            if user_text.strip():
                cleaned = normalize_and_mask(user_text)
                with st.expander("Show normalized text", expanded=False):
                    st.code(cleaned)
                X_single = vec.transform([cleaned])
                prob = float(clf.predict_proba(X_single)[:, 1][0])
                pred_label = pos_label if prob >= threshold else neg_label
                st.success(f"Prediction: {pred_label}  |  spam-prob = {prob:.4f}  (threshold = {threshold:.2f})")

                # Probability bar (0..1) with threshold marker
                fig_g, ax_g = plt.subplots(figsize=(6, 0.6))
                ax_g.barh([0], [prob], color="#d62728" if pred_label == pos_label else "#1f77b4")
                ax_g.axvline(threshold, color="black", linestyle="--", linewidth=1)
                ax_g.set_xlim(0, 1)
                ax_g.set_yticks([])
                ax_g.set_xlabel("spam probability")
                ax_g.text(min(prob + 0.02, 0.98), 0, f"{prob:.2f}", va="center")
                st.pyplot(fig_g)
            else:
                st.info("Please enter a non-empty message.")

    else:
        st.info("Model artifacts not found in 'models/'. Train the model first to enable performance plots.")


def main():
    page = st.sidebar.radio("Page", ["Demo", "Visualizations"]) 
    if page == "Demo":
        demo_page()
    else:
        viz_page()


if __name__ == "__main__":
    main()
