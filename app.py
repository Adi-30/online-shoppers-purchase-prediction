"""
app.py
------
Interactive Streamlit front-end for the Online Shoppers Purchasing Intention
classifiers. It loads the five pre-trained pipelines from ``model/`` and scores
an uploaded (or the bundled) test CSV WITHOUT any retraining.

Required features implemented:
    * CSV test-data upload (plus a one-click bundled sample)
    * Model-selection dropdown
    * Display of the six evaluation metrics
    * Confusion matrix and classification report
    * A side-by-side comparison of all five models on the same test data
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import data_utils as du

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"
CLASS_LABELS = ["No purchase", "Purchase"]

st.set_page_config(
    page_title="Online Shoppers Intention — Classifier Studio",
    page_icon="🛒",
    layout="wide",
)

# --- lightweight custom styling (keeps the stock template look off the page) --
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1200px;}
      .headline {font-size: 2.0rem; font-weight: 700; margin-bottom: .1rem;}
      .subtle {color: #6b7280; font-size: .95rem; margin-bottom: 1.1rem;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      .pill {display:inline-block; padding:.15rem .6rem; border-radius:999px;
             background:#eef2ff; color:#3730a3; font-size:.8rem; margin-right:.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Load every saved pipeline once per session."""
    loaded = {}
    for name, filename in du.MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_data(show_spinner=False)
def read_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TEST)


@st.cache_data(show_spinner=False)
def score_one(df: pd.DataFrame, model_name: str) -> dict:
    """Predictions + metrics for a single model on the given test frame."""
    models = load_models()
    pipeline = models[model_name]
    features = du.normalize_features(df[du.FEATURE_COLUMNS])
    y_true = du.coerce_target(df[du.TARGET_COLUMN]).to_numpy()
    y_pred = pipeline.predict(features)
    y_score = du.model_scores(pipeline, features)
    metrics = du.compute_metrics(y_true, y_pred, y_score)
    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics}


@st.cache_data(show_spinner=False)
def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics for every model, one row each."""
    rows = []
    for name in load_models():
        rows.append({"Model": name, **score_one(df, name)["metrics"]})
    table = pd.DataFrame(rows).set_index("Model")
    return table[du.METRIC_ORDER]


def confusion_figure(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


# =============================== Header =======================================
st.markdown('<div class="headline">🛒 Online Shoppers Purchasing Intention</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Five pre-trained classifiers scored live on your test '
    'data — no retraining. Predicting whether a browsing session ends in a '
    'purchase (<code>Revenue</code>).</div>',
    unsafe_allow_html=True,
)

models = load_models()
if not models:
    st.error("No trained models found in `model/`. Run `python train_models.py` first.")
    st.stop()

# =============================== Sidebar ======================================
with st.sidebar:
    st.header("1 · Test data")
    source = st.radio(
        "Choose a data source",
        ["Use bundled sample test set", "Upload my own test CSV"],
        help="The CSV must contain all 17 feature columns plus the Revenue target.",
    )

    uploaded = None
    if source == "Upload my own test CSV":
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])

    st.header("2 · Model")
    chosen_model = st.selectbox("Select a classifier", list(models.keys()))

    with st.expander("Required columns"):
        st.caption("Features:")
        st.code(", ".join(du.FEATURE_COLUMNS), language="text")
        st.caption(f"Target: {du.TARGET_COLUMN}")

# =============================== Load data ====================================
if source == "Upload my own test CSV":
    if uploaded is None:
        st.info("⬅️ Upload a test CSV in the sidebar, or switch to the bundled sample.")
        st.stop()
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that CSV: {exc}")
        st.stop()
else:
    data = read_sample()

absent = du.missing_columns(data)
if absent:
    st.error("The uploaded file is missing required column(s): " + ", ".join(absent))
    st.stop()

# --- data summary -------------------------------------------------------------
target_counts = du.coerce_target(data[du.TARGET_COLUMN]).value_counts().sort_index()
non_buy = int(target_counts.get(0, 0))
buy = int(target_counts.get(1, 0))
total = non_buy + buy

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows in test set", f"{total:,}")
c2.metric("Purchases", f"{buy:,}")
c3.metric("Non-purchases", f"{non_buy:,}")
c4.metric("Purchase rate", f"{(buy / total * 100):.1f}%" if total else "—")

with st.expander("Preview the test data"):
    st.dataframe(data.head(20), width="stretch")

st.divider()

# =============================== Tabs =========================================
single_tab, compare_tab = st.tabs(["🔍 Selected model", "📊 Compare all models"])

with single_tab:
    st.subheader(chosen_model)
    outcome = score_one(data, chosen_model)
    metrics = outcome["metrics"]

    m_cols = st.columns(6)
    for col, key in zip(m_cols, du.METRIC_ORDER):
        value = metrics[key]
        col.metric(key, "—" if pd.isna(value) else f"{value:.4f}")

    left, right = st.columns([1, 1.2])
    with left:
        st.pyplot(confusion_figure(outcome["y_true"], outcome["y_pred"]))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            outcome["y_true"],
            outcome["y_pred"],
            labels=[0, 1],
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, width="stretch")

with compare_tab:
    st.subheader("All five classifiers on this test data")
    table = score_all(data)

    st.dataframe(
        table.style.format("{:.4f}", na_rep="—").highlight_max(axis=0, color="#dcfce7"),
        width="stretch",
    )
    st.caption("Green cell = best score in that column.")

    metric_to_plot = st.selectbox("Chart a metric", du.METRIC_ORDER, index=4)  # F1 default
    chart_df = table[[metric_to_plot]].sort_values(metric_to_plot, ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(chart_df.index, chart_df[metric_to_plot], color="#6366f1")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_to_plot)
    ax.set_title(f"{metric_to_plot} by model")
    for y, v in enumerate(chart_df[metric_to_plot]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption(
    "Models: Logistic Regression · Decision Tree · K-Nearest Neighbors · "
    "Gaussian Naive Bayes · SVM (RBF). Dataset: UCI Online Shoppers Purchasing "
    "Intention. Metrics computed on uploaded test data only."
)
