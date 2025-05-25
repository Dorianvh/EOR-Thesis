import requests
import pandas as pd
from datetime import datetime

def get_day_ahead_prices(bidding_zone: str = 'NL', start_date: str = '2025-01-01', end_date: str = '2025-01-31', save_to_csv: bool = True):
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
        df.to_csv(f"data/{filename}", index=False)
        print(f"Data saved to {filename}")

    return df


def get_forecast_energy(production_type: str = 'solar', country: str = 'DE', forecast_type: str = 'current',
                        save_to_csv: bool = True):
    """
    Fetch forecasted power generation for a given source from the API and return a DataFrame.
    """
    # Make the API request
    url = 'https://api.energy-charts.info/public_power_forecast'
    params = {
        'country': country.lower(),
        'production_type': production_type.lower(),
        'forecast_type': forecast_type.lower()
    }
    response = requests.get(url, params=params)

    # Handle non-JSON or error responses
    if response.status_code != 200:
        raise ValueError(f"Request failed with status {response.status_code}: {response.text}")

    try:
        data = response.json()
    except ValueError:
        raise ValueError("Failed to decode JSON. Response was:\n" + response.text)

    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp_unix': data['unix_seconds'],
        f'{production_type}_forecast_mw': data['forecast_values']
    })

    # Add human-readable timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s')

    # Reorder columns
    df = df[['timestamp', 'timestamp_unix', f'{production_type}_forecast_mw']]

    if save_to_csv:
        filename = f"forecast_{production_type}_{country}_{forecast_type}.csv"
        df.to_csv(f"data/{filename}", index=False)
        print(f"Data saved to {filename}")

    return df


df = get_forecast_energy(production_type='solar', country='DE', forecast_type='current', save_to_csv=True)
print(df)
