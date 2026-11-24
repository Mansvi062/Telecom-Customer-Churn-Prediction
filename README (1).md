# Telecom Customer Churn Prediction and Analysis

A machine learning project that analyzes telecom customer data and predicts which customers are likely to leave (churn). It applies regression, classification, clustering, dimensionality reduction, a neural network and an ensemble model to the well-known Telco Customer Churn dataset.

## Dataset

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Records:** 7,043 customers
- **Columns:** 21 (customer ID, demographics, services subscribed, contract and billing details, tenure, charges, and the churn label)
- **Target:** `Churn` (Yes / No). About 26.5% of customers churned, so the classes are imbalanced.

## Project Workflow

1. **Data loading and inspection:** shape, data types, missing values, duplicates, unique values.
2. **Data cleaning:** converted `TotalCharges` to numeric and filled the 11 blank values.
3. **Exploratory data analysis:** churn distribution, churn by contract type, distributions of monthly charges and tenure, correlation heatmap.
4. **Regression (predicting `TotalCharges`):** Simple Linear, Multiple Linear and Polynomial Regression, evaluated with MAE, MSE, RMSE and R².
5. **Classification (predicting `Churn`):** Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, MLP (neural network) and Random Forest, evaluated with accuracy, precision, recall, F1, and AUC / log loss for the probabilistic models.
6. **Clustering:** K-Means (with elbow method) and Hierarchical Agglomerative Clustering on tenure, monthly charges and total charges.
7. **Dimensionality reduction:** PCA to 2 components for visualization.
8. **Model comparison:** results table and accuracy bar chart.

## Techniques Used

| Category | Methods |
|---|---|
| Regression | Simple Linear, Multiple Linear, Polynomial (degree 2) |
| Classification | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM (linear), MLP, Random Forest |
| Clustering | K-Means, Agglomerative (average linkage) |
| Dimensionality reduction | PCA |
| Preprocessing | One-hot encoding, standard scaling, stratified train/test split (80/20) |

## Results

Classification results on the 20% test set (`random_state=42`):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 |
| Random Forest | 0.786 | 0.624 | 0.492 | 0.550 |
| MLP | 0.752 | 0.534 | 0.505 | 0.519 |
| KNN | 0.747 | 0.525 | 0.500 | 0.512 |
| Decision Tree | 0.742 | 0.514 | 0.495 | 0.504 |
| Naive Bayes | 0.656 | 0.427 | 0.866 | 0.572 |

- **Best accuracy and F1:** Logistic Regression.
- **Best recall:** Naive Bayes (it catches the most churners, at the cost of more false alarms).
- **Multiple Linear Regression** predicting `TotalCharges` from `tenure` and `MonthlyCharges` reaches an R² of about 0.89. This is high largely because `TotalCharges` is roughly `tenure × MonthlyCharges`.

Exact values may vary slightly with library versions.

## Notes and Limitations

- The dataset is imbalanced, so accuracy alone can be misleading. Precision, recall and F1 give a fuller picture.
- The 11 missing `TotalCharges` values are filled with the median.
- Hierarchical clustering is run on the first 500 records to keep it fast.
- Models use default or lightly set hyperparameters, with no extensive tuning.

## Getting Started

### Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run

```bash
jupyter notebook Telecom_prediction.ipynb
```

Keep `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder as the notebook.

## Repository Structure

```
├── Telecom_prediction.ipynb               # Full analysis and models
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
└── README.md
```

## Possible Future Improvements

- Rank models by F1, recall or AUC instead of accuracy
- Handle class imbalance (class weights, SMOTE, threshold tuning)
- Cross-validation and hyperparameter tuning
- Feature importance analysis for business insights

## License

This project is for educational purposes.
