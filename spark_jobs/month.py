import pandas as pd

# Load CSV
df = pd.read_csv("data/raw/weather_meteo_by_airport.csv")

print(df.head())

# Convert to datetime
df["time"] = pd.to_datetime(df["time"])

# Unique month numbers
print(df["time"].dt.month.unique())

# Unique month names
print(df["time"].dt.month_name().unique())

# Unique dates count
print(df["time"].nunique())