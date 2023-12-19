# README for Bankruptcy Prediction Model

## Overview
This Python codebase provides a comprehensive framework for predicting corporate bankruptcy using a variety of machine learning models. It includes data preprocessing, feature engineering, model training, and evaluation phases. The approach uses financial ratios and other predictors to forecast bankruptcy events.

## Libraries
The script relies on several Python libraries:
- Pandas and NumPy for data manipulation.
- Statsmodels and Scikit-learn for statistical modeling and machine learning.
- Sklearn-survival for survival analysis in the context of bankruptcy prediction.
- Imbalanced-learn for handling imbalanced datasets.
- Matplotlib for data visualization.
- SciPy for additional statistical functions.
- XGBoost and LightGBM for gradient boosting models.

## Data Preprocessing and Feature Engineering
- `compute_financial_ratios`: Function to calculate financial ratios from input DataFrame.
- Data Cleaning: Processes raw bankruptcy data and financial data from 'BR1964_2019.csv' and 'funda_2022.csv' files.
- Merges data from different sources and prepares it for analysis.

## Models
- **Logistic Regression**: Traditional logistic regression model with feature selection and balancing using SMOTE.
- **Lasso Logistic Regression**: Logistic regression with L1 regularization.
- **Ridge Logistic Regression**: Logistic regression with L2 regularization.
- **K-Nearest Neighbors**: A simple instance-based learning algorithm.
- **Random Forest**: An ensemble method using a collection of decision tree classifiers.
- **Survival Random Forest**: An adaptation of the random forest for survival analysis.
- **XGBoost**: An implementation of gradient boosted decision trees.
- **LightGBM**: A fast, distributed, high-performance gradient boosting framework.
- **Artificial Neural Network**: A deep learning model built using Keras.

Each model follows a similar structure of data preparation, fitting, and evaluation.

## Evaluation
- Uses various metrics like accuracy, precision, recall, and AUC-ROC for model assessment.
- Implements cross-validation and grid search techniques for model tuning.
- Compares different models based on misclassification rates.

## Sentiment Analysis
- Extracts sections from Form 10-K filings.
- Uses FinBERT, a BERT-based model fine-tuned for financial sentiment analysis.
- Computes sentiment scores for each section and evaluates overall sentiment related to bankruptcy risk.

## Additional Notes
- The codebase is designed for extensibility and allows for the integration of additional models or features.
- Custom functions and methods are used extensively for data handling and model evaluation.

## Usage
1. Install required libraries: `pip install -r requirements.txt`.
2. Run the main script to preprocess data, train models, and evaluate: `python main.py`.
3. For sentiment analysis, ensure access to relevant 10-K filings and run the sentiment analysis module.

## Data Requirements
- The model requires historical financial data and bankruptcy records.
- Form 10-K filings for sentiment analysis.

## Acknowledgements
- The methodology and features are inspired by contemporary research in bankruptcy prediction.
- The codebase utilizes publicly available Python libraries and FinBERT model for sentiment analysis.
