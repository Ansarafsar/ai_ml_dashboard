import pandas as pd
import numpy as np
import streamlit as st
import requests
import altair as alt
import tensorflow as tf
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Retail Analytics Dashboard", page_icon="📈", layout="wide")

# Configuration
DATA_PATH = "data/pp_df_data.csv"
MODEL_DIR = "models"
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "your-api-key")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Model and scaler paths
MODEL_PATHS = {
    "Linear Regression": f"{MODEL_DIR}/linear_regression_model.pkl",
    "XGBoost": f"{MODEL_DIR}/xgboost_model.pkl",
    "Random Forest": f"{MODEL_DIR}/random_forest_model.pkl",
    "ARIMA": f"{MODEL_DIR}/arima_model.pkl",
    "LSTM": f"{MODEL_DIR}/lstm_model.h5"
}
SCALER_PATHS = {
    "Linear Regression": f"{MODEL_DIR}/scaler_lr.pkl",
    "XGBoost": f"{MODEL_DIR}/scaler_xgb.pkl",
    "Random Forest": f"{MODEL_DIR}/scaler_lr.pkl"  # Shared scaler
}

# Column aliases for BI queries
COLUMN_ALIASES = {
    "sales": "Sales", "profit": "Profit", "quantity": "Quantity", "discount": "Discount",
    "clv": "CLV", "age": "Customer Age", "fulfillment": "Fulfillment Time",
    "popularity": "Product Popularity", "discount impact": "Discount Impact",
    "year": "Year", "region": "Region", "state": "State", "category": "Category of Goods",
    "subcategory": "Sub-Category", "product": "Product Name", "customer": "Customer Name",
    "id": "Customer ID", "order": "Order ID", "ship": "Ship Mode"
}

# Utility functions
def load_data():
    """Load preprocessed retail sales data."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("Error: pp_df_data.csv not found in data/—please upload it.")
        st.stop()

def load_models_and_scalers():
    """Load ML models and scalers from models/ directory."""
    models = {}
    scalers = {}
    for name, path in MODEL_PATHS.items():
        try:
            if path.endswith('.pkl'):
                models[name] = pickle.load(open(path, "rb"))
            elif path.endswith('.h5'):
                models[name] = tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            st.warning(f"Failed to load model {name}: {str(e)}")
    for name, path in SCALER_PATHS.items():
        try:
            scalers[name] = pickle.load(open(path, "rb"))
        except Exception as e:
            st.warning(f"Failed to load scaler for {name}: {str(e)}")
    return models, scalers

def find_closest_row(data, quantity, discount, fulfillment_time):
    """Find the closest row in data based on input features."""
    data['distance'] = np.sqrt(
        (data['Quantity'] - quantity) ** 2 +
        (data['Discount'] - discount) ** 2 +
        (data['Fulfillment Time'] - fulfillment_time) ** 2
    )
    return data.loc[data['distance'].idxmin()]

def preprocess_input(row, scaler):
    """Preprocess input features for ML models."""
    features = ['Quantity', 'Discount', 'Fulfillment Time', 'Customer Age', 'CLV',
                'Discount Impact', 'Product Popularity', 'Year', 'Postal Code']
    input_features = row[features].values.reshape(1, -1)
    return scaler.transform(input_features)

def preprocess_lstm_input(quantity, discount, fulfillment_time, timesteps=6):
    """Preprocess input for LSTM model."""
    input_features = np.array([[quantity, discount, fulfillment_time]] * timesteps)
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(input_features)
    return scaled_input.reshape(1, timesteps, 3), scaler

def process_query(query, df, models, scalers, inputs=None):
    """Process BI query or ML prediction using LLM."""
    query = query.lower().strip()
    action = "predict" if inputs else ("top" if "top" in query else "summary")
    metric = None
    region = None
    n = 5

    # Parse query
    words = query.split()
    for i, word in enumerate(words):
        if word.isdigit():
            n = int(word)
        if 'in' in words[i:i+2] and i+2 < len(words):
            region = words[i+2].capitalize()
        for alias, col in COLUMN_ALIASES.items():
            if alias in query and col in df.columns:
                metric = col
                break
    metric = metric or "Sales"

    # Filter data
    filtered_df = df.copy()
    if region and region in df['Region'].unique():
        filtered_df = filtered_df[filtered_df['Region'] == region]
    if filtered_df.empty:
        return f"No data found for {region}."

    # Handle action
    context = ""
    if action == "predict" and inputs:
        model_choice = inputs.get("model")
        if model_choice not in models:
            return f"Model {model_choice} not available."
        try:
            if model_choice in ["Linear Regression", "XGBoost", "Random Forest"]:
                closest_row = find_closest_row(filtered_df, inputs["quantity"], inputs["discount"], inputs["fulfillment_time"])
                input_features = preprocess_input(closest_row, scalers[model_choice])
                prediction = models[model_choice].predict(input_features)[0]
            elif model_choice == "ARIMA":
                prediction = models["ARIMA"].forecast(steps=inputs.get("months", 1))[-1]
            elif model_choice == "LSTM":
                input_lstm, scaler = preprocess_lstm_input(inputs["quantity"], inputs["discount"], inputs["fulfillment_time"])
                prediction_scaled = models["LSTM"].predict(input_lstm, verbose=0)[0][0]
                prediction = scaler.inverse_transform([[prediction_scaled, 0, 0]])[0][0]
            context = f"Predicted Sales: ₹{prediction:,.2f}\nInputs: {inputs}"
        except Exception as e:
            return f"Prediction failed: {str(e)}"
    elif action == "top":
        top_items = filtered_df.sort_values(metric, ascending=False).head(n)
        if top_items[metric].isna().all():
            return f"No valid {metric} data found."
        context = top_items.to_csv(index=False)
    else:
        context = filtered_df.describe().to_csv() + "\nSample:\n" + filtered_df.head(5).to_csv(index=False)

    # LLM query
    history_context = "\n".join(
        f"Q: {chat['query']}\nA: {chat['response']}"
        for chat in st.session_state.chat_history[-3:]
    ) if st.session_state.chat_history else "No prior context."

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "You're a professional BI assistant—deliver clear, concise insights based on data or predictions."},
            {"role": "user", "content": f"History:\n{history_context}\nData/Prediction:\n{context}\nQuery: {query}\nProvide actionable insights."}
        ]
    }
    try:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            st.session_state.chat_history.append({"query": query, "response": answer})
            return answer
    except Exception as e:
        error_msg = f"Failed to fetch insights: {str(e)}"
        st.session_state.chat_history.append({"query": query, "response": error_msg})
        return error_msg

# Load data and models
df = load_data()
models, scalers = load_models_and_scalers()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# UI Layout
st.title("Retail Analytics Dashboard")
st.markdown("Predict sales, uncover insights, and visualize trends—all in one place.")

# Three-column layout
col1, col2, col3 = st.columns([2, 1, 1])

# BI Query (Left)
with col1:
    st.subheader("Business Insights")
    with st.form(key='query_form', clear_on_submit=True):
        query = st.text_input("Ask about your data", placeholder="e.g., 'Top 5 sales in East', 'Summary of profit'")
        submit_query = st.form_submit_button("Get Insights")

    if submit_query and query:
        response = process_query(query, df, models, scalers)
        st.success("Insight Generated!")
        st.markdown(f"**Answer:** {response}")

    st.subheader("Insight History")
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history[::-1][:5]):
            with st.expander(f"Q: {chat['query']} (#{len(st.session_state.chat_history)-i})"):
                st.write(f"A: {chat['response']}")
    else:
        st.info("No insights yet—ask away!")

# ML Prediction (Middle)
with col2:
    st.subheader("Sales Prediction")
    with st.form(key='predict_form'):
        model_choice = st.selectbox("Model", list(models.keys()))
        quantity = st.slider("Quantity", 1, 10, 1)
        discount = st.slider("Discount", 0.0, 1.0, 0.1, step=0.01)
        fulfillment_time = st.slider("Fulfillment Time (days)", 1, 10, 1)
        region = st.selectbox("Region", df['Region'].unique())
        months = st.slider("Months to Forecast", 1, 12, 1) if model_choice == "ARIMA" else 1
        submit_predict = st.form_submit_button("Predict")

    if submit_predict:
        inputs = {
            "model": model_choice, "quantity": quantity, "discount": discount,
            "fulfillment_time": fulfillment_time, "region": region, "months": months
        }
        response = process_query(f"Predict sales in {region}", df, models, scalers, inputs)
        st.success("Prediction Generated!")
        st.markdown(f"**Result:** {response}")

        # Quick viz for ML models
        if model_choice in ["Linear Regression", "XGBoost", "Random Forest"]:
            quantity_values = np.linspace(1, 10, 5)
            predictions = []
            for q in quantity_values:
                closest_row = find_closest_row(df, q, discount, fulfillment_time)
                input_features = preprocess_input(closest_row, scalers[model_choice])
                predictions.append(models[model_choice].predict(input_features)[0])
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(quantity_values.astype(str), predictions, color="#1E90FF")
            ax.set_xlabel("Quantity")
            ax.set_ylabel("Sales (₹)")
            ax.set_title(f"{model_choice} Forecast")
            st.pyplot(fig)

# Dashboard Stats (Right)
with col3:
    st.subheader("Data Snapshot")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Regions", df['Region'].nunique())
    st.metric("Total Sales", f"₹{df['Sales'].sum():,.2f}")
    st.metric("Average Profit", f"₹{df['Profit'].mean():,.2f}")

    # Dynamic chart
    if st.session_state.chat_history:
        last_query = st.session_state.chat_history[-1]['query']
        metric = next((COLUMN_ALIASES[alias] for alias in COLUMN_ALIASES if alias in last_query.lower()), 'Sales')
        top_5 = df.nlargest(5, metric)[['Product Name', metric]]
        chart = alt.Chart(top_5).mark_bar().encode(
            x=alt.X('Product Name', sort=None),
            y=metric,
            tooltip=['Product Name', metric]
        ).properties(width=200, height=200, title=f"Top 5 {metric}")
        st.altair_chart(chart)

# Footer
st.markdown(f"© 2025 Retail Analytics Dashboard | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
