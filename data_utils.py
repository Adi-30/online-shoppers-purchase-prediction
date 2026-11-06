"""
data_utils.py
-------------
Shared helpers used by both the training script (train_models.py) and the
Streamlit app (app.py). Keeping the column definitions, type-normalisation and
metric calculation in one place guarantees that the app evaluates an uploaded
test file exactly the way the models were trained and scored.

Nothing in here is pickled with the models — the saved pipelines contain only
standard scikit-learn transformers — so loading a model never depends on this
module being import-compatible. It is only used to prepare a DataFrame before
calling pipeline.predict().
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# --- Column layout of the Online Shoppers Purchasing Intention dataset ---------
TARGET_COLUMN = "Revenue"

# The two genuinely multi-level categorical columns that need one-hot encoding.
CATEGORICAL_COLUMNS = ["Month", "VisitorType"]

# The full set of 17 input feature columns, in their original order.
FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Pretty labels + the metric order used in every table / report.
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# Friendly names mapped to the model files saved under model/.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

_TRUE_TOKENS = {"true", "1", "yes", "y", "t"}
_FALSE_TOKENS = {"false", "0", "no", "n", "f"}


def _to_binary(series: pd.Series) -> pd.Series:
    """Coerce a bool / numeric / text yes-no column into clean 0/1 integers."""
    if series.dtype == bool:
        return series.astype("int64")
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype("int64")
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(lambda v: 1 if v in _TRUE_TOKENS else (0 if v in _FALSE_TOKENS else None))
    )
    return mapped.astype("int64")


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with dtypes the fitted pipelines expect:
    categorical columns forced to string, the Weekend flag forced to 0/1.
    Any numeric feature is left untouched. This is applied before every
    predict() call so training-time and app-time inputs are identical.
    """
    prepared = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype(str)
    if "Weekend" in prepared.columns:
        prepared["Weekend"] = _to_binary(prepared["Weekend"])
    return prepared


def coerce_target(series: pd.Series) -> pd.Series:
    """Turn the Revenue column (True/False, 0/1, yes/no...) into 0/1 integers."""
    return _to_binary(series)


def missing_columns(frame: pd.DataFrame) -> list[str]:
    """List any required feature/target columns absent from an uploaded file."""
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    return [c for c in required if c not in frame.columns]


def model_scores(estimator, feature_frame: pd.DataFrame):
    """
    Return a 1-D array of positive-class scores for ROC-AUC.
    Uses predicted probabilities when available, otherwise the SVM's
    signed distance to the decision boundary.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(feature_frame)[:, 1]
    return estimator.decision_function(feature_frame)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """The six evaluation metrics required by the assignment, as a dict.

    AUC needs both classes present in y_true; if an uploaded file happens to
    contain only one class it is reported as NaN rather than raising.
    """
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
