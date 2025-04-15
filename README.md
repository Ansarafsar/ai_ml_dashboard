# Retail Sales Prediction Dashboard

![Retail Analytics Dashboard](screenshots/dashboard.png)

A comprehensive retail analytics solution combining **machine learning (ML)** and **business intelligence (BI)** to predict sales and uncover insights from retail data. This project features exploratory data analysis (EDA), model training (Linear Regression, XGBoost, Random Forest, ARIMA, LSTM), and an interactive Streamlit dashboard integrating ML predictions with LLM-powered BI queries via OpenRouter API.

Built with Python, this project showcases end-to-end data science skills—data preprocessing, feature engineering, model development, and deployment—using a Kaggle retail sales dataset (100k records, India stores, 2019-2023).

## Features

- **Exploratory Data Analysis (EDA)**:
  - Data cleaning, feature engineering (e.g., CLV, discount impact).
  - Visualizations: sales trends, customer segmentation, discount-profit analysis.
- **Machine Learning Models**:
  - Linear Regression, XGBoost, Random Forest for regression tasks.
  - ARIMA and LSTM for time-series forecasting.
  - Metrics: MSE, R²; visualizations for feature importance and forecasts.
- **Interactive Dashboard**:
  - **ML Predictions**: Adjust sliders for Quantity, Discount, etc., to predict sales.
  - **BI Insights**: Query data (e.g., "Top 5 sales in East") using LLM (Mistral-7B via OpenRouter).
  - **Visualizations**: Dynamic bar charts, top-N metrics, data snapshots.
- **Deployment**: Streamlit app, runnable locally or via Colab with Ngrok.

## Repository Structure

Retail-Sales-Prediction-Dashboard/
├── data/
│   └── pp_df_data.csv              # Preprocessed retail sales data
├── notebooks/
│   ├── pp_fe_eda.ipynb             # Data preprocessing, feature engineering, EDA
│   ├── ml_model_training.ipynb     # Train and save ML models
│   └── llm_ml_integrated_dashboard.ipynb  # Build and launch Streamlit dashboard
├── models/
│   ├── linear_regression_model.pkl # Trained Linear Regression model
│   ├── scaler_lr.pkl               # Scaler for Linear Regression
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── scaler_xgb.pkl              # Scaler for XGBoost
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   ├── arima_model.pkl             # Trained ARIMA model
│   ├── lstm_model.h5               # Trained LSTM model
├── app.py                          # Streamlit dashboard code
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

## Prerequisites

- **Python**: 3.8+
- **Dependencies**: Listed in `requirements.txt`
- **Accounts**:
  - [OpenRouter](https://openrouter.ai) for LLM API key (free tier available).
  - [Ngrok](https://ngrok.com) for public URL (free account sufficient).
- **Dataset**: Kaggle retail sales data (provided as `pp_df_data.csv` after preprocessing).

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Retail-Sales-Prediction-Dashboard.git
   cd Retail-Sales-Prediction-Dashboard
