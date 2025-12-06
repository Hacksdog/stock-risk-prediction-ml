Stock Risk Prediction ML

A reusable machine learning project that predicts future stock risk levels (Low, Medium, High) using historical stock data, Z-score volatility analysis, and multiple ML models.
The system is designed so any user can replace the stock ticker (e.g., GOOGL → AAPL, TSLA, BTC-USD) and instantly train a new model for their own stock.

 Features--------------

Automatic stock data download using yfinance

Feature engineering with Z-score and OHLCV metrics

Future risk prediction using label shifting

Multiple machine learning models:

KNN (Best-K search)

Decision Tree

Random Forest

SVM (rbf, sigmoid, poly)

Confusion matrix visualizations

Accuracy comparison

Saves trained models using joblib

Manual prediction script for user-input data

Project Files
1. finance_analysis.py — Model Training & Saving

This script trains all models, evaluates performance, finds the best K for KNN, applies scaling, generates confusion matrices, and saves the final models (knn_model.pkl, dt_risk_model.pkl, rf_risk_model.pkl).
It also uses label shifting so the model predicts tomorrow’s risk instead of today’s.

2. new_data_prediction.py — Manual Future Prediction

This script loads the saved model and allows you to enter your own custom values:

new_data = [[Close, High, Low, Open, Volume, Z_score]]


Example output:

Low Risk
Medium Risk
High Risk

 How to Use With Any Stock

To train on another stock (e.g., Apple), change this line in finance_analysis.py:

df = yf.download("AAPL", start="2025-01-01", end="2025-12-05")


Everything else works automatically — feature engineering, model training, evaluation, and saving.
 How to Run
1. Train Models
python finance_analysis.py

2. Predict Using Custom New Data
python new_data_prediction.py

 Risk Levels
Value	Meaning
0	Low Risk
1	Medium Risk
2	High Risk

The model predicts next-day risk (thanks to label shifting).

Installation

Install all required packages:

pip install pandas numpy yfinance scikit-learn matplotlib joblib

 Why This Project Is Useful

Works for any stock or crypto symbol

Professionally structured ML pipeline

Easy to customize

Includes model comparison + visualization

Perfect for learning finance + machine learning

Ready for deployment, automation, or API integration
