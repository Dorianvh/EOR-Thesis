import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocess_data(file_path):
    """Load and preprocess raw CSV into hourly DataFrame"""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp')
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('h').ffill()
    return df


def prepare_armax_data(df, endog_col, exog_cols, forecast_horizon=3, lags=1, train_split=0.8):
    # create lag features and target shift
    dfc = df.copy()
    for i in range(1, lags+1):
        dfc[f'{endog_col}_lag_{i}'] = dfc[endog_col].shift(i)
    dfc['target'] = dfc[endog_col].shift(-forecast_horizon)
    model_cols = ['target'] + exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags+1)]
    dfm = dfc[model_cols].dropna()
    split = int(len(dfm)*train_split)
    train = dfm.iloc[:split]
    test = dfm.iloc[split:]
    exog_feats = exog_cols + [f'{endog_col}_lag_{i}' for i in range(1, lags+1)]
    return train['target'], train[exog_feats], test['target'], test[exog_feats]


def train_armax_model(endog, exog, ar_order=1, ma_order=0):
    model = SARIMAX(endog, exog=exog, order=(ar_order,0,ma_order),
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_armax_model(model_fit, steps, exog):
    return model_fit.forecast(steps=steps, exog=exog)


def evaluate_armax_model(actual, forecast, plot=False):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(actual.index, actual, label='Actual')
        plt.plot(forecast.index, forecast, label='Forecast')
        plt.legend(); plt.show()
    return rmse


if __name__ == '__main__':
    df = preprocess_data('../data/combined_energy_data.csv')
    y_train, X_train, y_test, X_test = prepare_armax_data(
        df, 'ID1_price_eur_per_mwh',
        ['solar_forecast_mw','wind_onshore_forecast_mw',
         'wind_offshore_forecast_mw','DA_price_eur_per_mwh'],
        forecast_horizon=3, lags=0, train_split=0.5
    )
    fit = train_armax_model(y_train, X_train, ar_order=1, ma_order=1)
    preds = forecast_armax_model(fit, steps=len(y_test), exog=X_test)
    rmse = evaluate_armax_model(y_test, preds, plot=True)
    print(f"Model RMSE: {rmse:.2f} €/MWh")



