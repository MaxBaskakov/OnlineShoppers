# Online Shoppers Purchasing Intention Prediction

A binary classification project to predict whether an online visitor will complete a purchase (`Revenue`) based on their session browsing behavior.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python online_shoppers.py
```

*The script automatically handles data preprocessing, model tuning, evaluation, and saves performance plots to the `plots/` directory.*

## Project Structure

```text
Online-Shoppers/
│
├── plots/                  # Generated evaluation charts
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── FeatureImportances.png
│   └── BasicVSTuned.png
│
├── online_shoppers.py      # Main pipeline script
├── requirements.txt        # Project dependencies
├── .gitignore
└── README.md
```

## Data Preparation

To ensure robust evaluation and avoid data leakage, the dataset was split into **training, validation, and test sets (60/20/20)**. The preprocessing steps include:

* **Target Isolation:** `Revenue` separated from features prior to transformations.
* **Categorical Encoding:** Processed via `OneHotEncoder(handle_unknown="ignore")`.
* **Numerical Scaling:** Normalized using `MinMaxScaler`.
* **Boolean Conversion:** `Weekend` feature converted to numerical format.

*Note: All preprocessing objects are fitted strictly on the training set and applied to the validation and test sets.*

## Models & Hyperparameter Tuning

We evaluated **Random Forest** and **XGBoost** as baseline models, followed by optimization via `GridSearchCV` (5-fold cross-validation).

**F1-score was used as the optimization metric.**

### Optimized Parameters

* **XGBoost:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=1.0`
* **Random Forest:** `max_depth=None`, `min_samples_split=2`, `n_estimators=200`

### Validation Performance

| Model                  |  Accuracy  |  Precision |   Recall   |  F1-Score  |   ROC-AUC  |
| :--------------------- | :--------: | :--------: | :--------: | :--------: | :--------: |
| Random Forest          |   89.78%   |   72.44%   |   54.09%   |   61.93%   |   92.94%   |
| **XGBoost (Selected)** | **90.27%** | **71.25%** | **61.48%** | **66.01%** | **93.42%** |

*XGBoost was selected as the final model based on the validation results, with higher F1-score and Recall than Random Forest.*

## Final Test Results

The optimized XGBoost model was evaluated **exactly once** on the unseen test set to verify generalization:

* **Accuracy:** 90.02%
* **Precision:** 71.88%
* **Recall:** 58.75%
* **F1-score:** 64.66%
* **ROC-AUC:** 92.69%

## Evaluation & Artifacts

The execution script automatically generates and saves the following assets to `plots/`:

* **Confusion Matrix & ROC Curve:** Used to evaluate classification performance and class separation.
* **Feature Importance:** Shows which features were most important for the XGBoost model's predictions.
* **Baseline vs Tuned Model:** Side-by-side metric comparison to evaluate the effect of hyperparameter tuning.

## Prediction on New Data

The project includes a helper function that accepts new visitor data, applies the same preprocessing, and returns a prediction.

Example:

```text
Result for new data: Customer
Probability: 72.35%
```

## Tech Stack

* **Core:** Python, NumPy, Pandas
* **ML Pipeline:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib
