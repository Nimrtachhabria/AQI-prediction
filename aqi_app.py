import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load your trained Random Forest model
# --------------------------------------------------
MODEL_PATH = "./models/aqi_random_forest.pkl"

with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def fetch_air_quality_data(lat=24.8608, lon=67.0104):
    """Fetch 3-day AQI and weather data from Open-Meteo API for Karachi"""
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,"
        f"sulphur_dioxide,ozone,nitrogen_dioxide,european_aqi"
        f"&past_days=92&forecast_days=3"
    )
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        f"&past_days=92&forecast_days=3"
    )

    air_data = requests.get(air_url).json()
    weather_data = requests.get(weather_url).json()

    air_df = pd.DataFrame(air_data["hourly"])
    air_df["time"] = pd.to_datetime(air_df["time"])

    weather_df = pd.DataFrame(weather_data["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    df = pd.merge(air_df, weather_df, on="time", how="inner")
    df["city"] = "Karachi"

    # Add date/time features
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month

    df.ffill(inplace=True)
    return df

def predict_aqi(df, model):
    """Use the trained model to predict AQI"""
    feature_cols = [
        "pm2_5",
        "pm10",
        "ozone",
        "carbon_monoxide",
        "sulphur_dioxide",
        "nitrogen_dioxide",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "hour",
        "day",
        "month",
    ]
    X = df[feature_cols]
    preds = model.predict(X)
    df["Predicted_AQI"] = preds
    return df

def aqi_level(aqi):
    """Categorize AQI levels"""
    if aqi <= 50:
        return "Good "
    elif aqi <= 100:
        return "Moderate "
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups "
    elif aqi <= 200:
        return "Unhealthy "
    elif aqi <= 300:
        return "Very Unhealthy "
    else:
        return "Hazardous 🚨"

# --------------------------------------------------
# Streamlit App UI
# --------------------------------------------------
st.set_page_config(page_title="AQI Predictor", layout="wide")

st.title("🌍  AQI Predictor – Karachi")
st.write("Predicting the **Air Quality Index (AQI)** for the next 3 days using live weather and pollutant data.")

# Fetch data
with st.spinner("Fetching latest air quality and weather data..."):
    data = fetch_air_quality_data()

# Predict AQI
with st.spinner("Running model predictions..."):
    predictions = predict_aqi(data, model)

# --------------------------------------------------
# Display current conditions
# --------------------------------------------------
st.subheader("📊 Current Air Quality Overview")

latest = predictions.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Current European AQI", f"{latest['european_aqi']:.1f}")
col2.metric("Predicted AQI", f"{latest['Predicted_AQI']:.1f}")
col3.metric("Condition", aqi_level(latest['Predicted_AQI']))

# --------------------------------------------------
# Plot forecast chart
# --------------------------------------------------
st.subheader("📈 3-Day AQI Forecast")
current_time = datetime.utcnow()
past_data = predictions[predictions["time"] <= current_time]
future_data = predictions[predictions["time"] > current_time]
start_date = current_time - timedelta(days=5)
end_date = current_time + timedelta(days=3)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(past_data["time"], past_data["european_aqi"], label="Actual AQI (Past)", color="skyblue")
ax.plot(future_data["time"], future_data["Predicted_AQI"], label="Forecasted AQI (Next 3 Days)", color="red", linewidth=2)
ax.set_xlim(start_date, end_date)  
ax.set_xlabel("Time")
ax.set_ylabel("AQI")
ax.set_title("3-Day AQI Forecast (Beyond Today)")
ax.legend()
st.pyplot(fig)


# --------------------------------------------------
# Show data table
# --------------------------------------------------
st.subheader("🗂️ Data Preview")
st.dataframe(predictions.tail(20))

# --------------------------------------------------
# AQI Alerts
# --------------------------------------------------
high_risk = predictions[predictions["Predicted_AQI"] > 150]
if not high_risk.empty:
    st.error("⚠️ Warning: Unhealthy air quality expected in the next 3 days. Limit outdoor activities.")
else:
    st.success("✅ Air quality is expected to remain moderate or good.")

# --------------------------------------------------
# Model Info
# --------------------------------------------------
st.sidebar.header("Model Info")
st.sidebar.write("**Algorithm:** Random Forest Regressor")
st.sidebar.write("**Trained on:** Historical AQI + weather data from Open-Meteo")
st.sidebar.write("**Performance (Training):**")
st.sidebar.write("- RMSE: 3.83")
st.sidebar.write("- MAE: 2.20")
st.sidebar.write("- R²: 0.94")
