# Credit Card Fraud Detection

This repository contains a project on credit card fraud detection. The goal of this project is to build a model that can accurately detect fraudulent credit card transactions. Various machine learning algorithms were employed to achieve this, including Logistic Regression, Random Forest Classifier, and XGBoost.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)

## Introduction

Credit card fraud detection is a crucial task for financial institutions to prevent significant financial losses. This project utilizes machine learning techniques to detect fraudulent transactions based on historical data.

## Data

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) available on Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.

## Preprocessing

The following preprocessing steps were performed on the dataset:
- **Visualization**: Explored the distribution of features and identified any patterns or anomalies.
- **Cleaning**: Checked for missing values and handled them appropriately.
- **Undersampling**: Addressed class imbalance by undersampling the majority class to create a balanced dataset.

## Modeling

Three different machine learning models were used to predict fraudulent transactions:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost (Extreme Gradient Boosting)**

Hyperparameter tuning was performed using GridSearchCV to find the optimal parameters for each model.

## Evaluation

The models were evaluated based on the following metrics:
- **Accuracy**
- **F1 Score**
- **Recall**
- **Precision**

## Results

The performance of the models was compared based on the evaluation metrics. The following results were observed:

| Model                   | Accuracy | F1 Score | Recall | Precision |
|-------------------------|----------|----------|--------|-----------|
| Logistic Regression     | 0.95     | 0.93     | 0.88   | 0.98      |
| Random Forest Classifier| 0.96     | 0.95     | 0.90   | 0.99      |
| XGBoost                 | 0.96     | 0.94     | 0.90   | 0.97      |

## Conclusion

- Logistic Regression provides a baseline performance.
- XGBoost improves the performance  due to its advanced boosting techniques.
- Random Forest Classifier offers the best performance with its ensemble approach.

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```
5. Open `Credit_Card_Fraud_Detection.ipynb` and run the cells to see the results.

## Requirements

- Python 3.6+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

You can install the required packages using:
```bash
pip install -r requirements.txt
