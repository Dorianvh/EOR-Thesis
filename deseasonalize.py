from prophet import Prophet
import pandas as pd
import plotly
import matplotlib.pyplot as plt

df = pd.read_csv("data/day_ahead_prices_NL_2016-01-01_2024-12-31.csv")

# Rename the columns for Prophet
df_prophet = df.rename(columns={'timestamp': 'ds', 'price_eur_per_mwh': 'y'})

# Initialize and fit the Prophet model
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(df_prophet)

# Make forecast on the same data
forecast = model.predict(df_prophet)

# Compute total seasonal component
forecast['seasonal'] = forecast['daily'] + forecast['weekly'] + forecast['yearly']

# Merge deseasonalized data into the original DataFrame
df_prophet['deseasonalized'] = df_prophet['y'] - forecast['seasonal']

print(df_prophet)
model.plot_components(forecast)

plt.figure(figsize=(15, 6))

# Plot original
plt.plot(df_prophet['ds'], df_prophet['y'], label='Original', alpha=0.7)

# Plot deseasonalized
plt.plot(df_prophet['ds'], df_prophet['deseasonalized'], label='Deseasonalized', alpha=0.7)

plt.title('Original vs. Deseasonalized Day-Ahead Prices')
plt.xlabel('Date')
plt.ylabel('Price (€/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()