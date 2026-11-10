# Data Science and Machine Learning Projects

An end-to-end machine learning project on a retail banking customer dataset (8,662 customers, 22 attributes). One shared, cleaned dataset feeds three modeling tasks: **churn classification**, **customer lifetime value (CLV) regression** and **customer segmentation**.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0)

## Workflow

```
data/raw/dataset.csv
        │
        ▼
01 · Preprocessing ──► data/processed/dataset_final.csv
        │                        │
        │            ┌───────────┴───────────┐
        ▼            ▼                       ▼
01 · Classification  02 · Regression         03 · Clustering
   (Churn)              (CLV_Continuous)        (Segments)
```

> Run **notebook 01 first**. It cleans the raw data and writes `dataset_final.csv`, which notebooks 02 and 03 read.

## Notebooks

| # | Notebook | Task | Models / techniques | View |
|---|----------|------|---------------------|------|
| 01 | [Preprocessing & Classification](notebooks/01-preprocessing-and-classification.ipynb) | Data cleaning + churn prediction | Logistic Regression, k-NN, Decision Tree, XGBoost, GridSearchCV | [HTML](notebooks/01-preprocessing-and-classification.html) |
| 02 | [Regression](notebooks/02-regression.ipynb) | Predict customer lifetime value | Linear, Polynomial (deg. 2–3), Ridge/Lasso, Regression Tree, Random Forest | [HTML](notebooks/02-regression.html) |
| 03 | [Clustering](notebooks/03-clustering.ipynb) | Customer segmentation | k-Means, Elbow method, Silhouette score, cluster profiling | [HTML](notebooks/03-clustering.html) |

## 1 · Data preprocessing

The raw data contains deliberate quality issues. The cleaning pipeline handles them as follows:

- **Inconsistent categories:** `CGeography` had 11 spellings (`France`, `Fr`, `SPAIN`, `DE`, …). These were standardized to 3 values.
- **Impossible values:** ages outside 18–100 (min −1, max 999), negative salaries and last logins older than the account itself were set to `NaN`.
- **Missing values:** median imputation for numeric columns and mode imputation for categorical columns.
- **Outliers:** winsorized at the 1st and 99th percentiles.
- **Feature engineering:** one-hot encoding (`drop_first=True`), `log1p` on skewed transaction features and removal of the redundant `Days_Since_Onboarding` column.
- **No data leakage:** scaling is done *after* the train/test split and fitted on training data only.

## 2 · Churn classification

| Model | Test accuracy | Notes |
|-------|---------------|-------|
| Logistic Regression | 92.79% | Recall on churners only 2.3% (accuracy paradox) |
| k-Nearest Neighbors | 92.21% | Recall 4.7% |
| Decision Tree (default) | 88.34% | Best baseline recall (32.8%), but overfits (train 100%) |
| Decision Tree (tuned) | 93.25% | `max_depth=4`; train/test gap reduced from 11.7% to 0.5% |
| **XGBoost (tuned)** | **93.83%** | Best CV accuracy (94.28%), feature importance by weight and gain |

## 3 · CLV regression

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Multiple Linear Regression | 0.9697 | 935.01 |
| Polynomial (degree 2) | 0.9692 | 942.52 |
| Polynomial (degree 3) | 0.9621 | 1046.74 |
| Lasso (α = 10) | 0.9697 | 935.29 |
| Regression Tree (tuned) | 0.9685 | 953.66 |
| **Random Forest (tuned)** | **0.9836** | **688.59** |

Higher polynomial degrees overfit: degree 3 expands the 19 features to 1,539. The Random Forest captures non-linear interactions that the linear models miss.

## 4 · Customer segmentation

k-Means with **k = 3** on 19 standardized features (silhouette score 0.073, so the clusters overlap):

| Cluster | Profile | Avg. balance | Avg. CLV | Recommendation |
|---------|---------|--------------|----------|----------------|
| 0 | High-balance, non-German, fewest products | $122,431 | $17,131 | Cross-sell additional products |
| 1 | Low-balance, non-German, most products | $6,758 | $8,638 | Capture payroll and grow deposits |
| 2 | German customers, high balance | $118,874 | $17,319 | Premium retention program |

Age, salary and credit score are almost identical across clusters. Balance, product count and geography are what separate them.

## Repository structure

```
.
├── data/
│   ├── raw/dataset.csv                 # original dataset
│   └── processed/dataset_final.csv     # output of notebook 01
├── notebooks/
│   ├── 01-preprocessing-and-classification.ipynb / .html
│   ├── 02-regression.ipynb / .html
│   └── 03-clustering.ipynb / .html
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/TelmanPrivateProjects/MachineLearning.git
cd MachineLearning

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab notebooks/
```

Run the notebooks in order (01 → 02 → 03). Data paths are relative to the `notebooks/` folder.

> **Reproducibility:** `requirements.txt` pins pandas 3.0, scikit-learn 1.8 and xgboost < 3.3. With these versions the notebooks reproduce the reported results exactly. Newer versions give slightly different tuned hyperparameters and scores.

To see the results without installing anything, open the `.html` files in `notebooks/`.
