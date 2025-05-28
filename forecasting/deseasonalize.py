from prophet import Prophet
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

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
plt.show()

# Extract the deseasonalized remainder
deseasonalized_remainder = df_prophet['deseasonalized'].dropna()

# Perform QQ computation
osm, osr = stats.probplot(deseasonalized_remainder, dist="norm", fit=True)

# Extract quantiles and fit line
theoretical_quants = osm[0]
sample_quants = osm[1]
slope, intercept = osr[0], osr[1]

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the QQ points in your custom blue
ax.plot(theoretical_quants, sample_quants, 'o', color="#1f77b4", markersize=4)

# Plot the reference (normal) line
line_x = np.linspace(min(theoretical_quants), max(theoretical_quants), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, color='red', lw=2)

# Add title and labels
ax.set_title('QQ Plot of Deseasonalized Remainder')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()
