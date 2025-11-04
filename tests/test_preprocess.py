from pathlib import Path

from src.spam_pipeline.preprocess import preprocess_text, build_tfidf_vectorizer


def test_preprocess_text_basic():
    s = "Hello!!! This is a test -- 123."
    out = preprocess_text(s)
    # 123 與 hello 與 test 應被保留並以空白分隔
    parts = out.split()
    assert "hello" in parts
    assert "123" in parts
    assert "this" in parts


def test_tfidf_vectorizer_on_sample():
    corpus = [
        "Hello world, this is ham message",
        "Win money now spam offer",
        "hello friend this is another ham",
    ]
    corpus_clean = [preprocess_text(s) for s in corpus]
    vec, X = build_tfidf_vectorizer(corpus_clean, max_features=50, use_stop_words=False)
    assert X.shape[0] == len(corpus)
    vocab = vec.vocabulary_
    assert "hello" in vocab
    assert "spam" in vocab
    assert "money" in vocab
