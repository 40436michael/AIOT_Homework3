import re
from typing import Iterable, Tuple
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin

_DEFAULT_TOKEN_RE = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    """
    輕量文字清理與標準化：
    - to lower
    - 移除非字母數字字元（保留空白分隔）
    - collapse 多重空白
    回傳清理後的字串（適合給 sklearn 的 TfidfVectorizer）
    """
    if text is None:
        return ""
    s = str(text).lower()
    tokens = _DEFAULT_TOKEN_RE.findall(s)
    return " ".join(tokens)


def normalize_and_mask(text: str) -> str:
    """
    更完整的正規化與遮罩流程：
    - 將 URL 替換為 <URL>
    - 將 email 替換為 <EMAIL>
    - 將 phone 替換為 <PHONE>
    - 將剩餘數字序列替換為 <NUM>
    - 移除多餘標點並壓縮空白
    回傳已小寫並遮罩的字串
    """
    if text is None:
        return ""
    s = str(text)

    # URL
    s = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', s, flags=re.IGNORECASE)

    # Email
    s = re.sub(r'\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b', ' <EMAIL> ', s, flags=re.IGNORECASE)

    # Phone (loose pattern)
    s = re.sub(r'\b(?:\+?\d[\d\-\s]{6,}\d)\b', ' <PHONE> ', s)

    # Replace remaining digit sequences with <NUM>
    s = re.sub(r'\d+', ' <NUM> ', s)

    # Remove characters except word chars, angle brackets (to keep tags), and spaces
    s = re.sub(r"[^\w\s<>]", ' ', s)

    # collapse multiple whitespace and lower
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


class PreprocessTransformer(TransformerMixin):
    """
    sklearn-style transformer: 將文字欄位清理後傳遞給 vectorizer
    用法: Pipeline 中可放入此 transformer
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_text(x) for x in X]


def build_tfidf_vectorizer(
    corpus: Iterable[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
    use_stop_words: bool = True
):
    """
    建構並以 corpus 擬合 TF-IDF vectorizer，回傳 (vectorizer, X_tfidf)
    """
    stop_words = "english" if use_stop_words else None
    vect = TfidfVectorizer(
        preprocessor=lambda x: x,  # 文字已由 PreprocessTransformer 清理
        tokenizer=lambda x: x.split(),
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    X = vect.fit_transform(corpus)
    return vect, X


def save_vectorizer(vec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)
    return path


def load_vectorizer(path: Path):
    return joblib.load(path)
