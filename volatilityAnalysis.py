import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/day_ahead_prices_DE-LU_2018-01-01_2025-01-01.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate 24-hour (or 7-day) rolling volatility
df['rolling_volatility'] = df['price_eur_per_mwh'].rolling(window=24).std()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['timestamp'], df['rolling_volatility'], label='24h Rolling Volatility')

# Define events
events = {
    "Russia Invades Ukraine": "2022-02-24",
    "Nord Stream Sabotage": "2022-09-26",
    "Russia-Ukraine transit agreement ends": "2025-01-01",

}

# Annotate with arrows
y_max = df['rolling_volatility'].max()
y_positions = [y_max*0.9, y_max*0.8, y_max*0.7, y_max*0.6, y_max*0.5]

for i, (event, date) in enumerate(events.items()):
    date_obj = pd.to_datetime(date)

    ax.annotate(event,
                xy=(date_obj, df[df['timestamp'] == date_obj]['rolling_volatility'].values[0] if date_obj in df['timestamp'].values else y_max*0.3),
                xytext=(date_obj, y_positions[i]),
                arrowprops=dict(facecolor='red', arrowstyle="->"),
                fontsize=14,
                horizontalalignment='right')

# Final formatting
ax.set_title('Rolling Volatility of Electricity Prices with Key Events')
ax.set_xlabel('Time')
ax.set_ylabel('Volatility (EUR/MWh)')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()