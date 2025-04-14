import requests
import pandas as pd
from datetime import datetime

def get_day_ahead_prices(bidding_zone: str = 'NL', start_date: str = '2016-01-01', end_date: str = '2024-12-31', save_to_csv: bool = True):
    """
    Fetch day-ahead  prices from the API and return a DataFrame.
    """
    # Make the API request
    url = 'https://api.energy-charts.info/price'
    params = {
        'bzn': bidding_zone,
        'start': start_date,
        'end': end_date
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp_unix': data['unix_seconds'],
        'price_eur_per_mwh': data['price']
    })

    # Add human-readable timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s')

    # Reorder columns
    df = df[['timestamp', 'timestamp_unix', 'price_eur_per_mwh']]

    if save_to_csv:
        # Save to CSV
        filename = f"day_ahead_prices_{bidding_zone}_{start_date}_{end_date}.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    return df

get_day_ahead_prices()
