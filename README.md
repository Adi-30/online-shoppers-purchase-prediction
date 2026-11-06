# Online Shoppers Purchasing Intention — Classifier Studio

An end-to-end machine-learning classification project: five classifiers are
trained on the **UCI Online Shoppers Purchasing Intention** dataset, saved as
self-contained preprocessing + model pipelines, and served through an
interactive **Streamlit** web app that scores an uploaded test CSV **without
retraining**.

---

## a. Problem Statement

Online retailers see far more browsing sessions than actual purchases, and
knowing *which* sessions are likely to convert lets a business target offers,
remarketing and on-site nudges where they matter. The task here is a **binary
classification** problem: given behavioural and technical attributes of a single
browsing session (pages visited, time spent, Google-Analytics bounce/exit/page
values, month, visitor type, etc.), predict whether the session ends in a
purchase — the `Revenue` column (`True`/`False`).

Because only ~15.6% of sessions convert, the dataset is **class-imbalanced**,
which makes it a realistic test of how different algorithms behave when accuracy
alone can be misleading.

## b. Dataset Description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository ([dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)) |
| Task | Binary classification |
| Instances | 12,330 sessions (12,205 after removing 125 duplicate rows) |
| Features | 17 input features (10 numeric, 2 multi-level categorical, plus technical/indicator fields) |
| Target | `Revenue` — `True` (purchase) / `False` (no purchase) |
| Class balance | ~15.6% purchases / ~84.4% non-purchases |
| Missing values | None |

**Feature groups**

- **Behavioural (numeric):** `Administrative`, `Administrative_Duration`,
  `Informational`, `Informational_Duration`, `ProductRelated`,
  `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`,
  `SpecialDay`
- **Categorical:** `Month`, `VisitorType`
- **Technical / indicator:** `OperatingSystems`, `Browser`, `Region`,
  `TrafficType`, `Weekend`

**Preprocessing.** `Month` and `VisitorType` are one-hot encoded; numeric
features are standardised for the distance- and margin-based models (Logistic
Regression, KNN, SVM) and for Naive Bayes, while the Decision Tree is left
unscaled (trees are scale-invariant). All preprocessing is baked into each saved
pipeline, so a model consumes raw feature rows directly. The data is split
80/20 with **stratification** to preserve the class ratio, using
`random_state=42` for reproducibility.

## c. GitHub Repository Link

> **Repository:** https://github.com/Adi-30/online-shoppers-purchase-prediction
> **Live Streamlit app:** `<STREAMLIT_APP_URL>`

*(The Streamlit link is filled in once the app is deployed to Streamlit
Community Cloud.)*

## d. Models Used

All five classifiers are trained on the **same** dataset and split, and each is
evaluated on the held-out 20% test set (2,441 sessions) using six metrics.

### Evaluation metrics comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | **0.9022** | **0.7718** | 0.4162 | 0.5408 | 0.5142 |
| Decision Tree | 0.8599 | 0.7453 | 0.5498 | 0.5785 | 0.5638 | 0.4806 |
| K-Nearest Neighbors | 0.8787 | 0.8539 | 0.7337 | 0.3534 | 0.4770 | 0.4536 |
| Naive Bayes | 0.7108 | 0.8268 | 0.3284 | **0.8115** | 0.4676 | 0.3757 |
| **Support Vector Machine** | **0.8943** | 0.8798 | 0.7403 | 0.5000 | **0.5969** | **0.5524** |

*Bold = best score in that column. Metrics are reported on the positive
(`purchase`) class at the default 0.5 decision threshold.*

> **Why accuracy looks high everywhere:** with only ~15.6% buyers, a model can
> score ~0.84 accuracy by predicting "no purchase" almost every time. The
> **F1**, **MCC** and **Recall** columns are what actually separate the models,
> which is exactly why the assignment asks for all six.

### Observations on model performance

| ML Model | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong, well-generalised baseline (train 0.883 ≈ test 0.889, so no overfitting). **Best AUC (0.9022)** and **best precision (0.7718)**: it ranks sessions by purchase likelihood better than any other model and rarely raises a false alarm. Its weakness is recall (0.42) — at the default threshold it is conservative and misses ~58% of actual buyers. Ideal when you need a reliable probability score. |
| **Decision Tree** | Fully-grown tree **overfits**: 1.00 training accuracy vs 0.86 on test. It has the most balanced precision/recall of the group (0.55 / 0.58) but the **weakest AUC (0.7453)**, because an unpruned tree produces poor, step-like probability estimates. Would improve with depth limiting, pruning, or an ensemble (Random Forest / Gradient Boosting). |
| **K-Nearest Neighbors** | Reasonable precision (0.734) but **low recall (0.353)**. In this imbalanced, mixed-type feature space the 15-neighbour vote is dominated by the majority (non-buyer) class, so many buyers are missed. Middling ranking ability (AUC 0.854). Sensitive to scaling (handled) and to the choice of *k*. |
| **Naive Bayes** | The opposite profile: **highest recall (0.8115)** but **lowest precision (0.328)** and **lowest accuracy (0.711)**. The Gaussian feature-independence assumption is violated — page counts and durations are strongly correlated — so it over-predicts "purchase," flagging most buyers at the cost of many false positives. Useful only when missing a potential buyer is far costlier than a false alarm. |
| **Support Vector Machine** | **Best overall balance**: top **accuracy (0.8943)**, **F1 (0.5969)** and **MCC (0.5524)**. The RBF kernel captures non-linear structure the linear models cannot, and it copes with the imbalance better than KNN or Naive Bayes at the default threshold. Only marginally behind Logistic Regression on pure ranking (AUC). |
| **Overall — Winner for this dataset?** | **Support Vector Machine (RBF).** It leads on the three metrics that best summarise imbalanced binary performance — Accuracy, F1 and MCC. Two runner-up notes: if the goal is **ranking / probability scoring** rather than hard labels, **Logistic Regression** wins (best AUC + precision and no overfitting); if the priority is **catching every possible buyer** (maximising recall), **Naive Bayes** is the choice. For a single all-round classifier, SVM. |

---

## Project Structure

```
online-shoppers-purchase-prediction/
├── app.py                 # Streamlit web app (loads saved models, scores test data)
├── train_models.py        # Trains all 5 pipelines, writes model/ + test_data.csv
├── data_utils.py          # Shared columns, type-normalisation, metric helpers
├── requirements.txt       # Pinned dependencies (match the saved-model versions)
├── test_data.csv          # 20% hold-out split used by the app (features + target)
├── data/
│   └── online_shoppers_intention.csv   # Full source dataset (for reproducibility)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json       # Held-out metrics + dataset summary
```

## How to Run Locally

```bash
# 1. Install dependencies (ideally in a fresh virtual environment)
pip install -r requirements.txt

# 2. (Optional) retrain the models from scratch — regenerates model/ + test_data.csv
python train_models.py

# 3. Launch the web app
streamlit run app.py
```

## Streamlit App Features

- **Test-data upload (CSV)** — upload your own test file, or use the bundled
  `test_data.csv` sample with one click.
- **Model-selection dropdown** — pick any of the five trained classifiers.
- **Evaluation metrics** — all six metrics shown as live cards for the selected
  model, computed on the uploaded data.
- **Confusion matrix + classification report** — per-class breakdown for the
  selected model.
- **Compare-all view** — a single table and bar chart scoring all five models on
  the same test data, with the best score in each column highlighted.

The app **loads the saved pipelines and evaluates the uploaded test data without
retraining**, as required.

## Reproducibility & Methodology Notes

- Fixed `random_state=42` for the split and all stochastic estimators.
- Duplicate rows are dropped before splitting to avoid identical rows leaking
  across the train/test boundary.
- Dependency versions in `requirements.txt` are pinned to the exact versions
  used to create the `.joblib` files, so the saved models load identically in
  deployment (scikit-learn pickles are version-sensitive).
- Models are **not** tuned heavily — the aim is an honest, like-for-like
  comparison of the five algorithms with sensible defaults.

## Tech Stack

Python · scikit-learn · pandas · NumPy · Streamlit · matplotlib · seaborn ·
joblib

**Live Streamlit app:** 
`https://online-shoppers-purchase-prediction-tj35svftdaxaqjtqh4fvql.streamlit.app/`