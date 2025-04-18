# 𝐑𝐞𝐭𝐚𝐢𝐥𝐒𝐲𝐧𝐜 𝐀𝐈: 𝐌𝐋-𝐏𝐨𝐰𝐞𝐫𝐞𝐝 𝐒𝐚𝐥𝐞𝐬 𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐢𝐧𝐠 & 𝐁𝐈 𝐈𝐧𝐬𝐢𝐠𝐡𝐭𝐬

![RetailSync AI Dashboard](screenshots/dashboard.png)

RetailSync AI is a cutting-edge data science project that forecasts retail sales and delivers actionable business intelligence (BI) using a Kaggle dataset (100k records, India stores, 2019-2023). Built with Python, it combines **machine learning (ML)** models and **large language model (LLM)**-powered queries to empower retail decision-making. The project features exploratory data analysis (EDA), model training (Linear Regression, XGBoost, Random Forest, ARIMA, LSTM), and a dynamic Streamlit dashboard for real-time predictions and insights.

This repository showcases end-to-end skills—data preprocessing, feature engineering, ML pipelines, and interactive deployment—perfect for retail analytics innovation. A huge **thank you** to Abu Humza Khan for the Kaggle dataset ([Store Sales Data](https://www.kaggle.com/datasets/abuhumzakhan/store-data)), which fueled this project!

## Features

- **Exploratory Data Analysis (EDA)**:
  - Cleans and preprocesses retail sales data.
  - Engineers features like Customer Lifetime Value (CLV) and Discount Impact.
  - Visualizes trends, customer segments, and discount-profit correlations.
- **Machine Learning Models**:
  - Trains Linear Regression, XGBoost, Random Forest (R² ~0.5–0.7), ARIMA, and LSTM (R² ~0.7).
  - Evaluates with MSE, R², and visualizes feature importance and forecasts.
- **Interactive Streamlit Dashboard**:
  - **ML Predictions**: Sliders for Quantity, Discount, and more to forecast sales.
  - **BI Insights**: LLM-powered queries (Mistral-7B via OpenRouter) like “Top 5 sales in East.”
  - **Visualizations**: Dynamic Altair charts for top metrics and predictions.
- **Deployment**: Runs locally or via Google Colab with Ngrok for public access.
---
## Structure:
```
 RetailSync-AI/
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
```
---
## Prerequisites

**Install Dependencies**:
```
pip install -r requirements.txt
```
- **Python**: 3.8+
- **Dependencies**: Listed in `requirements.txt`
- **Accounts**:
  - [OpenRouter](https://openrouter.ai) for LLM API key (free tier available).
  - [Ngrok](https://ngrok.com) for public URL (free account sufficient).
- **Dataset**: [Kaggle retail sales data](https://www.kaggle.com/datasets/abuhumzakhan/store-data) (provided as `pp_df_data.csv` after preprocessing).

## Set Environment Variables:
Create a .env file or set secrets in Colab:
```
OPENROUTER_API_KEY=your-openrouter-key
NGROK_TOKEN=your-ngrok-token
```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Retail-Sales-Prediction-Dashboard.git
   cd Retail-Sales-Prediction-Dashboard
   ```
2. **Launch Dashboard**:
   ```
   streamlit run app.py
   ```
