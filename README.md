# AQI-prediction
AQI predictor is end to end machine learning  project  that predicts the air quality index for the next 3 days in karachi using real time weather and pollutant data  from open mateo api.

🚀 Project Overview

This project builds a complete pipeline for air quality forecasting:

Feature Pipeline – Fetches live and historical weather + pollutant data using Open-Meteo, processes it, and stores features in Hopsworks Feature Store every hour.

Training Pipeline – Trains a Random Forest model daily on updated feature data to predict future AQI values.

Streamlit Dashboard – Displays real-time AQI forecasts, trends, and health alerts interactively.

CI/CD Workflow – Managed with GitHub Actions to automatically trigger:

Hourly feature updates

Daily model retraining

🧠 Tech Stack

Python, Pandas, Scikit-learn, Joblib

Hopsworks Feature Store (for data management)

Streamlit (for dashboard visualization)

GitHub Actions (for CI/CD automation)

Open-Meteo API (for weather and pollution data)


⚡ Highlights

100% serverless pipeline

Real-time data ingestion and prediction

Automated retraining and monitoring

Transparent model results and AQI category alerts
