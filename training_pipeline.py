import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

# --- Connect to Hopsworks project ---
project = hopsworks.login()
fs = project.get_feature_store()

# --- Load historical data from Feature Store ---
feature_group = fs.get_feature_group(name="karachi_air_quality", version=1)
df = feature_group.read()

print(f"✅ Data loaded {len(df)} rows successfully from Feature Store")
print(df.head())

# --- Drop missing target rows ---
df = df.dropna(subset=["european_aqi"])

# --- Select Features and Target ---
target = "european_aqi"
features = [
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

X = df[features]
y = df[target]

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# --- Save model locally ---
os.makedirs("models", exist_ok=True)
model_path = "models/aqi_random_forest.pkl"
joblib.dump(model, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")
