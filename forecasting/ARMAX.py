import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
from deseasonalize import deseasonalize_data

# Load and preprocess data
df = pd.read_csv('../data/combined_energy_data.csv')
# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])

#Remove duplicates (keeps first occurrence)
df = df.drop_duplicates(subset='timestamp')

df.set_index('timestamp', inplace=True)
df = df.asfreq('h')  # Assuming hourly data

# If there are missing values after setting frequency, you may want to handle them
df = df.ffill()  # Forward fill any missing values created


def train_armax_forecast_with_plot(df, endog_col, exog_cols, forecast_horizon=3, lags=1, train_split=0.8, ar_order=1, ma_order=0):
    """
    Trains an ARMAX model and plots the forecast vs ground truth.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Create lagged endogenous features
    for i in range(1, lags + 1):
        df_copy[f'{endog_col}_lag_{i}'] = df_copy[endog_col].shift(i)

    # Shift target
    df_copy['target'] = df_copy[endog_col].shift(-forecast_horizon)

    # Drop NaNs
    model_cols = ['target'] + exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]
    df_model = df_copy[model_cols].dropna()

    # Train-test split
    split_idx = int(len(df_model) * train_split)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    # Exogenous inputs
    exog_train = train[exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]]
    exog_test = test[exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]]

    # Fit ARMAX with error handling
    try:
        model = SARIMAX(train['target'], exog=exog_train, order=(ar_order, 0, ma_order),
                       enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    except Exception as e:
        print(f"Error fitting ARMAX model: {e}")
        return None

    # Forecast
    pred = model_fit.forecast(steps=len(test), exog=exog_test)

    # RMSE
    rmse = np.sqrt(mean_squared_error(test['target'], pred))

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test['target'], label='Actual', alpha=0.7)
    plt.plot(test.index, pred, label='Forecast', alpha=0.7)
    plt.title(f'Forecast vs Actual (horizon = {forecast_horizon}, RMSE = {rmse:.2f} €/MWh)')
    plt.xlabel('Time')
    plt.ylabel('ID1 Price (€/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return rmse


# Store the RMSE value when running the function
rmse = train_armax_forecast_with_plot(
    df=df,
    endog_col='ID1_price_eur_per_mwh',
    exog_cols=['solar_forecast_mw', 'wind_onshore_forecast_mw', 'wind_offshore_forecast_mw', 'DA_price_eur_per_mwh'],
    forecast_horizon=3,
    lags=0,
    train_split=0.8,
    ar_order=1,
    ma_order=1
)

print(f"Model RMSE: {rmse:.2f} €/MWh")