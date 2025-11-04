from pathlib import Path
import pandas as pd

from src.spam_pipeline.ingest import load_dataset, validate_dataset


def test_load_and_validate(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("ham,Hello there\nspam,Win $$$ now\nham,How are you?\n")
    df = load_dataset(sample)
    ok, msg = validate_dataset(df)
    assert ok, msg
    assert list(df.columns) == ["label", "message"]
    assert df.shape[0] == 3
    assert set(df["label"].unique()) <= {"ham", "spam"}
