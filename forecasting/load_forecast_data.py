import pandas as pd

from utils.api_connector import *

start_date = '2023-01-01'
end_date = '2024-12-31'
country = 'DE'
forecast_type = 'day-ahead'

df = pd.read_csv('../data/ID1_prices_germany.csv')
# filter on start and end date
# rename columns to match the expected format
df = df.rename(columns={'Date': 'timestamp', 'ID1_price': 'ID1_price_eur_per_mwh'})
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]




solar_forecast = get_forecast_energy(production_type='solar', country=country, forecast_type=forecast_type, start_date=start_date, end_date=end_date, save_to_csv=False)
wind_onshore_forecast = get_forecast_energy(production_type='wind_onshore', country=country, forecast_type=forecast_type, start_date=start_date, end_date=end_date, save_to_csv=False)
wind_offshore_forecast = get_forecast_energy(production_type='wind_offshore', country=country, forecast_type=forecast_type, start_date=start_date, end_date=end_date, save_to_csv=False)
load_forecast = get_forecast_energy(production_type='load', country=country, forecast_type=forecast_type, start_date=start_date, end_date=end_date, save_to_csv=False)

#combine quarter hourly data into hourly data
def combine_quarter_hourly_to_hourly(df):
    """
    Combine quarter-hourly data into hourly data by adding the values of each quarter hour within an hour.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Resample to hourly frequency and sum the values
    hourly_df = df.resample('h').sum()
    # Reset index to have 'timestamp' as a column again
    hourly_df.reset_index(inplace=True)
    hourly_df.drop(columns=['timestamp_unix'], inplace=True)
    return hourly_df


solar_forecast = combine_quarter_hourly_to_hourly(solar_forecast)
wind_onshore_forecast = combine_quarter_hourly_to_hourly(wind_onshore_forecast)
wind_offshore_forecast = combine_quarter_hourly_to_hourly(wind_offshore_forecast)
load_forecast = combine_quarter_hourly_to_hourly(load_forecast)

da_prices = get_day_ahead_prices(bidding_zone='DE-LU', start_date=start_date, end_date=end_date, save_to_csv=False)

# merge all dataframes on timestamp dont keep the timestamp_unix column

df = df.merge(solar_forecast, on='timestamp', how='left')
df = df.merge(wind_onshore_forecast, on='timestamp', how='left')
df = df.merge(wind_offshore_forecast, on='timestamp', how='left')
df = df.merge(load_forecast, on='timestamp', how='left')
df = df.merge(da_prices, on='timestamp', how='left')

# Save the combined DataFrame to a CSV file
df.to_csv('../data/combined_energy_data.csv', index=False)


