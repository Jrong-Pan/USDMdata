
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "USDMDataAvg.csv"
RESULT_DIR = ROOT / "result"

LABEL_ORDER = ["0", "D0", "D1", "D2", "D3", "D4"]
LABEL_MAP = {s: i for i, s in enumerate(LABEL_ORDER)}


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        print("Place USDMDataAvg.csv under ./data/ (Dryad) and rerun.")
        sys.exit(1)

    print("Loading CSV (large file may take several minutes)...", flush=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, sorting by time...", flush=True)
    df = df.sort_values("time")
    train_df = df[df["time"] < 20180000].copy()
    test_df = df[df["time"] >= 20180000].copy()
    print(f"Train rows: {len(train_df):,}, test rows: {len(test_df):,}", flush=True)

    train_df["_y"] = train_df["drought"].map(LABEL_MAP)
    test_df["_y"] = test_df["drought"].map(LABEL_MAP)
    train_df = train_df[train_df["_y"].notna()].copy()
    test_df = test_df[test_df["_y"].notna()].copy()

    drop_cols = ["Unnamed: 0", "time", "grid", "drought", "_y"]
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].copy()
    X_test = X_test[common].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0.0)
    X_test = X_test.fillna(med).fillna(0.0)

    def _finite_df(x: pd.DataFrame) -> pd.DataFrame:
        arr = np.nan_to_num(x.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(arr, columns=x.columns, index=x.index)

    print("Coercing numeric types and cleaning NaN/Inf...", flush=True)
    X_train = _finite_df(X_train)
    X_test = _finite_df(X_test)

    y_train = train_df["_y"].to_numpy(dtype=int)
    y_test = test_df["_y"].to_numpy(dtype=int)
    return X_train, X_test, y_train, y_test


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = list(range(len(LABEL_ORDER)))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    )
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    out: dict[str, float] = {
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
    }
    for i, name in enumerate(LABEL_ORDER):
        out[f"Recall_{name}"] = float(recalls[i])
    return out


def save_cm(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in LABEL_ORDER], columns=[f"pred_{x}" for x in LABEL_ORDER])
    cm_df.to_csv(RESULT_DIR / f"week12_cm_{name}.csv")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("Week12 comparison — this run can take 10–40+ minutes on the full Dryad CSV.", flush=True)
    X_train, X_test, y_train, y_test = load_split()
    print(f"Features: {X_train.shape[1]}, X_train shape: {X_train.shape}", flush=True)

    runs: list[tuple[str, object]] = []

    runs.append(
        (
            "rf_default",
            RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1),
        )
    )
    runs.append(
        (
            "rf_class_weight_balanced",
            RandomForestClassifier(
                n_estimators=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        )
    )
    runs.append(
        (
            "lr_class_weight_balanced",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
    )

    xgb_default = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(LABEL_ORDER),
        random_state=42,
        n_jobs=-1,
    )
    runs.append(("xgb_default", xgb_default))

    xgb_sw = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(LABEL_ORDER),
        random_state=42,
        n_jobs=-1,
    )

    rows: list[dict[str, object]] = []

    for name, model in runs:
        print(f"Training: {name} ...", flush=True)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        block = metrics_block(y_test, y_pred)
        block["Model"] = name
        rows.append(block)
        save_cm(name, y_test, y_pred)

    print("Training: xgb_sample_weight_balanced ...", flush=True)
    sw = compute_sample_weight("balanced", y_train)
    xgb_sw.fit(X_train, y_train, sample_weight=sw)
    y_pred_sw = np.asarray(xgb_sw.predict(X_test), dtype=int)
    block_sw = metrics_block(y_test, y_pred_sw)
    block_sw["Model"] = "xgb_sample_weight_balanced"
    rows.append(block_sw)
    save_cm("xgb_sample_weight_balanced", y_test, y_pred_sw)

    out = pd.DataFrame(rows)
    front = ["Model", "Accuracy", "Macro_F1", "Weighted_F1"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + sorted(rest)]
    out_path = RESULT_DIR / "week12_model_comparison.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
