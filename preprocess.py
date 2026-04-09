"""
preprocess.py
-------------
Loads and cleans news datasets for fake news detection.

Supported datasets:
  • WELFake  — WELFake_Dataset.csv  (72,134 articles)
  • ISOT     — True.csv + Fake.csv  (~44,898 articles)
  • Combined — WELFake + ISOT merged (~117,032 articles)

WELFake CSV columns : Unnamed: 0 | title | text | label  (1=fake, 0=real)
ISOT CSV columns    : title | text | subject | date
  True.csv → real (label=0)   Fake.csv → fake (label=1)

Run standalone to preview a dataset:
  python preprocess.py --dataset welfake
  python preprocess.py --dataset isot
  python preprocess.py --dataset combined
"""

import argparse
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove URLs/HTML/digits/punctuation, lemmatize, drop stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"\d+", "", text)                      # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def combine_title_text(row) -> str:
    """Combine title + text into one field for richer features."""
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    text  = str(row.get("text",  "")) if pd.notna(row.get("text"))  else ""
    return f"{title} {text}".strip()


def _finalize(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Shared post-processing: combine fields, clean, drop empties, cast label."""
    # Work on an explicit copy so pandas never raises SettingWithCopyWarning
    df = df.copy()

    print(f"[INFO] Combining title and text …")
    df["combined"] = df.apply(combine_title_text, axis=1)

    print(f"[INFO] Cleaning text (lemmatization + stopword removal) …")
    df["combined"] = df["combined"].apply(clean_text)

    df = df[df["combined"].str.strip().astype(bool)].copy()
    df["label"] = df["label"].astype(int)

    print(f"[INFO] {source_tag} clean shape : {df.shape}")
    print(f"[INFO] Label dist  : {df['label'].value_counts().to_dict()}  (1=fake, 0=real)")
    return df[["combined", "label"]].reset_index(drop=True)


# ── WELFake Loader ─────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str = "WELFake_Dataset.csv") -> pd.DataFrame:
    """
    Load WELFake CSV, clean text, return DataFrame with columns ['combined', 'label'].
    Label convention: 1 = fake, 0 = real.
    """
    print(f"[INFO] Loading WELFake dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    print(f"[INFO] Raw shape: {df.shape}")

    df.dropna(subset=["label"], inplace=True)
    df = df[~(df["title"].isna() & df["text"].isna())]

    # WELFake on Kaggle ships with 1=fake, 0=real — matches our convention directly
    return _finalize(df, "WELFake")


# ── ISOT Loader ────────────────────────────────────────────────────────────────

def load_and_preprocess_isot(
    true_csv: str = "True.csv",
    fake_csv: str = "Fake.csv",
) -> pd.DataFrame:
    """
    Load the ISOT Fake News Dataset (University of Victoria).
    True.csv  → real news → label 0
    Fake.csv  → fake news → label 1

    ISOT columns: title | text | subject | date
    """
    print(f"[INFO] Loading ISOT dataset …")
    print(f"  Real articles : {true_csv}")
    print(f"  Fake articles : {fake_csv}")

    df_real = pd.read_csv(true_csv)
    df_fake = pd.read_csv(fake_csv)

    print(f"[INFO] ISOT raw — real: {len(df_real):,}  fake: {len(df_fake):,}")

    df_real["label"] = 0   # real
    df_fake["label"] = 1   # fake

    df = pd.concat([df_real, df_fake], ignore_index=True)

    # Drop rows missing both title and text
    df = df[~(df["title"].isna() & df["text"].isna())]
    df.dropna(subset=["label"], inplace=True)

    return _finalize(df, "ISOT")


# ── Combined Loader ────────────────────────────────────────────────────────────

def load_and_preprocess_combined(
    welfake_csv: str = "WELFake_Dataset.csv",
    isot_true_csv: str = "True.csv",
    isot_fake_csv: str = "Fake.csv",
) -> pd.DataFrame:
    """
    Merge WELFake + ISOT into one shuffled DataFrame.
    Both datasets use the same label convention (1=fake, 0=real) so they
    can be concatenated directly without any remapping.
    """
    print("=" * 55)
    print("  Loading WELFake …")
    print("=" * 55)
    df_welfake = load_and_preprocess(welfake_csv)

    print()
    print("=" * 55)
    print("  Loading ISOT …")
    print("=" * 55)
    df_isot = load_and_preprocess_isot(isot_true_csv, isot_fake_csv)

    print()
    df = pd.concat([df_welfake, df_isot], ignore_index=True)

    # Shuffle so the two sources are interleaved during training
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[INFO] Combined dataset shape : {df.shape}")
    print(f"[INFO] Label dist             : {df['label'].value_counts().to_dict()}  (1=fake, 0=real)")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess fake-news datasets")
    parser.add_argument(
        "--dataset", choices=["welfake", "isot", "combined"], default="welfake",
        help="Which dataset(s) to load (default: welfake)"
    )
    parser.add_argument("--csv",       default="WELFake_Dataset.csv", help="Path to WELFake CSV")
    parser.add_argument("--isot-true", default="True.csv",            help="Path to ISOT True.csv")
    parser.add_argument("--isot-fake", default="Fake.csv",            help="Path to ISOT Fake.csv")
    args = parser.parse_args()

    if args.dataset == "welfake":
        df = load_and_preprocess(args.csv)
        out = "welfake_clean.csv"
    elif args.dataset == "isot":
        df = load_and_preprocess_isot(args.isot_true, args.isot_fake)
        out = "isot_clean.csv"
    else:
        df = load_and_preprocess_combined(args.csv, args.isot_true, args.isot_fake)
        out = "combined_clean.csv"

    df.to_csv(out, index=False)
    print(f"[✓] Saved cleaned dataset to {out}")