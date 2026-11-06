"""
train_models.py
---------------
Trains the five required classifiers on the UCI Online Shoppers Purchasing
Intention dataset and saves each one as a single, self-contained pipeline
(preprocessing + estimator) under ``model/``.

Run from the project root:

    python train_models.py

Outputs
    model/<name>.joblib   one pipeline per classifier
    model/metrics.json    held-out test metrics + dataset summary
    test_data.csv         the 20% hold-out split, for the Streamlit app

Design notes
    * Each pipeline one-hot-encodes Month / VisitorType and (for the distance /
      coefficient based models) standardises the numeric columns, so a saved
      model consumes raw feature rows directly.
    * The split is stratified because the target is imbalanced (~15% buyers),
      and duplicate rows are dropped first to avoid identical rows leaking
      across the train/test boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_utils as du

RANDOM_STATE = 42
TEST_FRACTION = 0.20

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
METRICS_JSON = MODEL_DIR / "metrics.json"


def build_pipeline(estimator, scale_numeric: bool) -> Pipeline:
    """Wrap one estimator in a preprocessing + model pipeline."""
    numeric_columns = [c for c in du.FEATURE_COLUMNS if c not in du.CATEGORICAL_COLUMNS]
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), du.CATEGORICAL_COLUMNS),
            ("numeric", numeric_step, numeric_columns),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def model_zoo() -> dict[str, Pipeline]:
    """The five classifiers, each with preprocessing suited to it."""
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            scale_numeric=True,
        ),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            scale_numeric=False,  # trees are scale-invariant
        ),
        "K-Nearest Neighbors": build_pipeline(
            KNeighborsClassifier(n_neighbors=15),
            scale_numeric=True,  # distances need comparable scales
        ),
        "Naive Bayes": build_pipeline(
            GaussianNB(),
            scale_numeric=True,
        ),
        "Support Vector Machine": build_pipeline(
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            scale_numeric=True,  # RBF kernel is scale-sensitive
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before} rows; {len(raw)} remain after dropping "
          f"{before - len(raw)} duplicates.")

    target = du.coerce_target(raw[du.TARGET_COLUMN])
    features = raw[du.FEATURE_COLUMNS]

    feat_train, feat_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    # Save the hold-out split with ORIGINAL dtypes so it mirrors a real upload.
    hold_out = feat_test.copy()
    hold_out[du.TARGET_COLUMN] = raw.loc[feat_test.index, du.TARGET_COLUMN].values
    hold_out.to_csv(TEST_CSV, index=False)
    print(f"Wrote {TEST_CSV.name}: {len(hold_out)} rows "
          f"({int(y_test.sum())} buyers / {int((1 - y_test).sum())} non-buyers).")

    train_ready = du.normalize_features(feat_train)
    test_ready = du.normalize_features(feat_test)

    results = {}
    print("\n" + "=" * 78)
    print(f"{'Model':<24}{'Acc':>8}{'AUC':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'MCC':>8}")
    print("-" * 78)

    for name, pipeline in model_zoo().items():
        pipeline.fit(train_ready, y_train)

        y_pred = pipeline.predict(test_ready)
        y_score = du.model_scores(pipeline, test_ready)
        metrics = du.compute_metrics(y_test, y_pred, y_score)

        # Train accuracy is kept only to comment on over-/under-fitting later.
        train_accuracy = pipeline.score(train_ready, y_train)

        results[name] = {
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "train_accuracy": round(float(train_accuracy), 4),
        }

        slug = du.MODEL_FILES[name]
        joblib.dump(pipeline, MODEL_DIR / slug)

        print(f"{name:<24}"
              f"{metrics['Accuracy']:>8.4f}{metrics['AUC']:>8.4f}"
              f"{metrics['Precision']:>8.4f}{metrics['Recall']:>8.4f}"
              f"{metrics['F1']:>8.4f}{metrics['MCC']:>8.4f}")

    print("=" * 78)

    summary = {
        "dataset": "UCI Online Shoppers Purchasing Intention",
        "n_rows_after_dedup": int(len(raw)),
        "n_features": len(du.FEATURE_COLUMNS),
        "n_train": int(len(feat_train)),
        "n_test": int(len(feat_test)),
        "positive_rate": round(float(target.mean()), 4),
        "random_state": RANDOM_STATE,
        "results": results,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {len(results)} models + {METRICS_JSON.name} to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
