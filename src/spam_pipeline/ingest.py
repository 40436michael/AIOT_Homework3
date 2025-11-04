import argparse
from pathlib import Path
import requests
import pandas as pd
from typing import Tuple

RAW_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"


def download_dataset(dest_path: Path, force: bool = False) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        return dest_path
    resp = requests.get(RAW_URL, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest_path


def load_dataset(path: Path) -> pd.DataFrame:
    # Source file has no header: first col = label, second = message
    df = pd.read_csv(path, header=None, names=["label", "message"], encoding="latin-1")
    # Basic cleaning
    df = df.dropna(subset=["label", "message"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    if not {"label", "message"}.issubset(df.columns):
        return False, "missing required columns"
    if df[["label", "message"]].isnull().any().any():
        return False, "nulls found in required columns"
    allowed = {"ham", "spam"}
    if not set(df["label"].unique()).issubset(allowed):
        return False, f"unexpected labels: {set(df['label'].unique()) - allowed}"
    return True, "ok"


def save_processed(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download and validate SMS spam CSV")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/sms_spam_no_header.csv"))
    parser.add_argument("--force", action="store_true", help="redownload even if file exists")
    parser.add_argument("--save-processed", type=Path, default=Path("data/processed/sms_spam_clean.csv"))
    args = parser.parse_args()

    path = download_dataset(args.dest, force=args.force)
    print(f"Downloaded: {path}")
    df = load_dataset(path)
    ok, msg = validate_dataset(df)
    if not ok:
        print(f"Validation FAILED: {msg}")
        raise SystemExit(2)
    print(f"Validation PASSED: {msg}")
    saved = save_processed(df, args.save_processed)
    print(f"Processed dataset saved to: {saved}")


if __name__ == "__main__":
    main()
