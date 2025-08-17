# train/train_model.py
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------------------------
# Paths / CLI
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "merged_train_resources.csv"  # <— your file name inside train/
MODEL_DIR = ROOT / "app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Builds robust text & numeric features regardless of which raw columns are present."""
    df = df.copy()

    # --- Build combined_essays (robustly) ---
    text_candidates = [
        "combined_essays",
        "project_essay_1", "project_essay_2", "project_essay_3", "project_essay_4",
        "project_title",
        "project_resource_summary",
        "description",
    ]
    existing_text_cols = [c for c in text_candidates if c in df.columns]

    if existing_text_cols:
        df["combined_essays"] = (
            df[existing_text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        df["combined_essays"] = "No essay provided"

    df["combined_essays"] = df["combined_essays"].fillna("No essay provided")

    # Optional resource summary (not required by model)
    if "project_resource_summary" not in df.columns:
        df["project_resource_summary"] = "No description"
    else:
        df["project_resource_summary"] = df["project_resource_summary"].fillna("No description")

    # --- Numeric features (robust to strings/NaNs) ---
    price = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
    qty = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
    df["total_cost"] = price * qty

    prior = pd.to_numeric(df.get("teacher_number_of_previously_posted_projects", 0), errors="coerce").fillna(0)
    df["teacher_number_of_previously_posted_projects"] = prior
    df["teacher_experience"] = (prior > 0).astype(int)

    # Date features
    dt = pd.to_datetime(df.get("project_submitted_datetime"), errors="coerce")
    df["submit_month"] = dt.dt.month.fillna(0).astype(int)
    df["submit_dow"] = dt.dt.dayofweek.fillna(0).astype(int)

    # Resource line count per project (if multiple rows per project exist)
    if "id" in df.columns:
        df["num_resources"] = df.groupby("id")["id"].transform("count").astype(int)
    else:
        df["num_resources"] = 1

    # Essay length
    df["essay_length"] = df["combined_essays"].apply(lambda x: len(str(x).split()))

    return df


def build_pipeline(categorical_cols, numeric_cols, text_col):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
            ("txt", TfidfVectorizer(max_features=5000, stop_words="english"), text_col),
        ]
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    return pipe


def main(data_name: str):
    # Resolve CSV path
    data_path = ROOT / "train" / data_name
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {data_path}. Place it under the train/ folder or pass --data PATH")

    # Load & FE
    df = load_data(data_path)
    df = feature_engineering(df)

    # Feature lists
    categorical_cols = ["teacher_prefix", "school_state", "project_grade_category", "project_subject_categories"]
    numeric_cols = [
        "essay_length",
        "total_cost",
        "teacher_experience",
        "teacher_number_of_previously_posted_projects",
        "submit_month",
        "submit_dow",
        "num_resources",
    ]
    text_col = "combined_essays"

    # Ensure required columns exist (fill safe defaults)
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].astype(str).fillna("Unknown")

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "project_is_approved" not in df.columns:
        raise KeyError("Missing 'project_is_approved' (target) in dataset.")

    # Target & Features
    y = df["project_is_approved"].astype(int)
    X = df[categorical_cols + numeric_cols + [text_col]].copy()

    # Split (quick metrics; final model is fitted on all data)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = build_pipeline(categorical_cols, numeric_cols, text_col)

    # CV on train split (robust estimate)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV ROC-AUC: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")

    # Fit on all rows
    pipe.fit(X, y)

    # Persist
    model_path = MODEL_DIR / "model.joblib"
    meta_path = MODEL_DIR / "meta.json"
    joblib.dump(pipe, model_path)

    meta = {
        "version": "1.0.0",
        "model_type": "LogisticRegression+OHE+TFIDF",
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "text_col": text_col,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved meta  to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DonorsChoose approval model.")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="CSV file name under train/ (or relative path). Default: merged_train_resources.csv",
    )
    args = parser.parse_args()
    main(args.data)
