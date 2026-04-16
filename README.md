# Breast Cancer Classification — End-to-End ML Project

A complete, well-commented machine learning pipeline that classifies breast tumours as **malignant or benign** using the [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) bundled with scikit-learn.

---

## Project Overview

This notebook walks through every stage of a real ML project:

| Step | Description |
|------|-------------|
| 1 | Load & explore the data |
| 2 | Initial visualisation (feature histograms by diagnosis) |
| 3 | Stratified train / test split |
| 4 | Deep EDA on training data (UMAP, correlation heatmap, pairplot, boxplots) |
| 4b | **Feature engineering** — ratio-based derived features |
| 4c | **PCA** — explained variance & 2-D projection |
| 5 | Data preparation (imputation, encoding, scaling pipelines) |
| 6 | Train & compare four models with cross-validation |
| 7 | Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) + **learning curves** |
| 8 | Final evaluation (accuracy, AUC, confusion matrix, **PR curve**, **classification report**) |
| 9 | Model serialisation with `joblib` |

---

## New Features (vs. original notebook)

### 🔵 Support Vector Machine
An RBF-kernel SVM pipeline is trained and cross-validated alongside Logistic Regression, Decision Tree and Random Forest, giving a fourth data point for model selection.

### 📊 Model Comparison Chart
A horizontal bar chart shows mean cross-validated AUC ± standard deviation for all four models at a glance, making it easy to pick the best candidate for fine-tuning.

### ⚗️ Feature Engineering
Three interpretable ratio features are computed before the train/test split:
- `mean_perimeter_radius_ratio` — circularity proxy
- `mean_concavity_smoothness_ratio` — irregular vs. smooth boundary
- `mean_compactness_index` — area relative to perimeter squared

A quick cross-validation check measures whether these features move the needle for Logistic Regression.

### 🔍 PCA Analysis
- Cumulative explained-variance plot shows how many components capture 95% of variance.
- A 2-D PCA scatter plot complements the existing UMAP visualisation for a fuller picture of cluster separability.

### 📈 Learning Curves
Side-by-side learning curves for Logistic Regression and Random Forest diagnose bias vs. variance and show whether more training data would help.

### 🎯 Precision-Recall Curve
A PR curve with average-precision score is plotted on the test set — especially important in medical contexts where recall (catching malignant cases) matters more than raw accuracy.

### 📋 Full Classification Report
`sklearn.metrics.classification_report` prints per-class precision, recall, F1-score and support, providing richer context than accuracy alone.

---

## Dataset

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569
- **Features**: 30 real-valued features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each computed for mean, SE and worst)
- **Classes**: Malignant (mapped to 1) / Benign (mapped to 0)

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
umap-learn      # optional — used in the UMAP visualisation cell
joblib
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy umap-learn joblib
```

---

## Running the Notebook

### Locally (Jupyter)

```bash
jupyter notebook breast_cancer_classification_enhanced.ipynb
```

### Google Colab

Click the **Open in Colab** badge at the top of the notebook, then install `umap-learn` if you want the UMAP cell:

```python
!pip install umap-learn
```

---

## Key Results (typical run)

| Model | Mean 10-fold CV AUC |
|-------|-------------------|
| Logistic Regression | ~0.994 |
| Decision Tree | ~0.923 |
| Random Forest | ~0.993 |
| SVM (RBF) | ~0.997 |

> Exact values vary slightly with random state and scikit-learn version.

After GridSearchCV fine-tuning the Random Forest achieves **~97–99% accuracy** and **~0.99 AUC** on the held-out test set.

---

## Project Structure

```
.
├── breast_cancer_classification_enhanced.ipynb   # Main notebook
├── breast_cancer_classification_model.pkl        # Saved model (created when notebook is run)
└── README.md
```

---

## Notes & Caveats

- This is a **learning/demonstration project**, not a clinical tool.
- The Wisconsin dataset is clean and well-separated — real-world diagnostic data will be messier and require more rigorous validation.
- Always evaluate medical models with domain experts and appropriate statistical tests before any deployment consideration.
