import hopsworks
import pandas as pd
import requests
import datetime
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

# --- Connect to Hopsworks project ---
project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()

# --- Fetch AQI + weather data from Open-Meteo API ---
latitude = 24.8608  # Karachi latitude
longitude = 67.0104 # Karachi longitude

air_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8608&longitude=67.0104&hourly=european_aqi,pm10,pm2_5,carbon_monoxide,carbon_dioxide,sulphur_dioxide,ozone,nitrogen_dioxide&past_days=92&forecast_days=3&domains=cams_global"
   
)

air_response = requests.get(air_url).json()

air_df = pd.DataFrame(air_response["hourly"])
air_df["time"] = pd.to_datetime(air_df["time"])

weather_url = (
  "https://api.open-meteo.com/v1/forecast?latitude=24.8608&longitude=67.0104&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&past_days=92&forecast_days=3"
)    

weather_response = requests.get(weather_url).json()
weather_df = pd.DataFrame(weather_response["hourly"])
weather_df["time"] = pd.to_datetime(weather_df["time"])


df = pd.merge(air_df, weather_df, on="time", how="inner")
df["city"] = "karachi"
print(df.head())

# --- Add time-based features ---
df["hour"] = df["time"].dt.hour.astype(int)
df["day"] = df["time"].dt.day.astype(int)
df["month"] = df["time"].dt.month.astype(int)

# --- Handle missing data ---

df.ffill(inplace=True)



# --- Create (or get) a feature group ---
feature_group = fs.get_or_create_feature_group(
    name="karachi_air_quality",
    version=1,
    primary_key=["time"],
    event_time="time",
    description="Real-time air quality and weather features for karachi"
)

# --- Insert data into feature store ---
feature_group.insert(df)

print("✅ Data successfully inserted into Hopsworks Feature Store!")
print(f"Rows added: {len(df)}")
