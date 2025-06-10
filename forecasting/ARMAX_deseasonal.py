import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
from deseasonalize import deseasonalize_data

# Load and preprocess data
df = pd.read_csv('../data/combined_energy_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.drop_duplicates(subset='timestamp')
df.set_index('timestamp', inplace=True)
df = df.asfreq('h')  # Assuming hourly data
df = df.ffill()

df.reset_index(inplace=True)

# Apply deseasonalization
df = deseasonalize_data(df, 'timestamp', 'ID1_price_eur_per_mwh', plot=False)

def train_armax_forecast_deseasonalized(df, endog_col, exog_cols, forecast_horizon=3, lags=1, train_split=0.8, ar_order=1, ma_order=0):
    df_copy = df.copy()

    # Create lagged endogenous features
    for i in range(1, lags + 1):
        df_copy[f'{endog_col}_lag_{i}'] = df_copy[endog_col].shift(i)

    # Shift target (forecasting the remainder)
    df_copy['target_remainder'] = df_copy[endog_col].shift(-forecast_horizon)

    # Ensure all components are present for reconstruction
    model_cols = ['target_remainder', 'seasonal_component', 'trend_component', 'ID1_price_eur_per_mwh'] + exog_cols + \
                 [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]
    df_model = df_copy[model_cols].dropna()

    # Train-test split
    split_idx = int(len(df_model) * train_split)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]

    # Exogenous input
    exog_train = train[exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]]
    exog_test = test[exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags + 1)]]

    # Fit ARMAX model on the remainder
    model = SARIMAX(train['target_remainder'], exog=exog_train, order=(ar_order, 0, ma_order),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast deseasonalized remainder
    pred_remainder = model_fit.forecast(steps=len(test), exog=exog_test)

    # Reconstruct full price
    forecast_reconstructed = pred_remainder + test['seasonal_component'].values + test['trend_component'].values
    actual_price = test['ID1_price_eur_per_mwh'].values

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(actual_price, forecast_reconstructed))

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(test.index, actual_price, label='Actual ID1 Price', alpha=0.7)
    plt.plot(test.index, forecast_reconstructed, label='Forecasted Reconstructed Price', alpha=0.7)
    plt.title(f'Reconstructed ID1 Price Forecast (horizon = {forecast_horizon}, RMSE = {rmse:.2f} €/MWh)')
    plt.xlabel('Time')
    plt.ylabel('ID1 Price (€/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return rmse

# Run the model
rmse = train_armax_forecast_deseasonalized(
    df=df,
    endog_col='deseasonalized_remainder',
    exog_cols=['solar_forecast_mw', 'wind_onshore_forecast_mw', 'wind_offshore_forecast_mw', 'DA_price_eur_per_mwh'],
    forecast_horizon=3,
    lags=0,
    train_split=0.8,
    ar_order=1,
    ma_order=1
)

print(f"Reconstructed Model RMSE: {rmse:.2f} €/MWh")
