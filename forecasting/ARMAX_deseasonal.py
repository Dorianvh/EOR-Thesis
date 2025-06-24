import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
from deseasonalize import deseasonalize_data
from outlier_detection import remove_outliers
import os

import itertools
from datetime import datetime

# Configuration - All parameters defined in one place
CONFIG = {
    # Input and output
    'input_file': '../data/combined_energy_data.csv',
    'output_file': '../data/ID1_price_forecast_1hour.csv',



    # Target variable
    'target_col': 'ID1_price_eur_per_mwh',
    'date_col': 'timestamp',

    # Model parameters
    'forecast_horizon': 24,  # forecast hours ahead
    'lags': 2,              # number of autoregressive lags
    'ar_order': 0,          # AR order for ARMAX
    'ma_order': 0,          # MA order for ARMAX
    'train_split': 0.5,     # portion of data for training

    # Grid search parameters
    'grid_search': False,    # whether to perform grid search
    'grid_params': {
        'lags': [0, 1, 2, 3, 4],
        'ar_order': [0, 1, 2, 3, 4],
        'ma_order': [0, 1, 2, 3, 4]
    },

    # Exogenous variables
    'exog_cols': ['solar_forecast_mw', 'wind_onshore_forecast_mw',
                  'wind_offshore_forecast_mw', 'DA_price_eur_per_mwh'],

    # Visualization
    'plot_results': True,  # whether to show plots
}

# Preprocess to hourly series
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset='timestamp')
    df = remove_outliers(df, 'timestamp', CONFIG['target_col'], threshold=10, plot=CONFIG['plot_results'])

    df.set_index('timestamp', inplace=True)
    df = df.asfreq('h').ffill()

    # remove outliers


    # only save 2023 data
    #df = df[df.index.year == 2023]

    return df

# Deseasonalize series using existing routine
def deseasonalize_series(df, date_col, y_col, plot=False):
    df_in = df.reset_index() if df.index.name == date_col else df.copy()
    df_out = deseasonalize_data(df_in, date_col, y_col, plot=plot)
    df_out.set_index(date_col, inplace=True)
    return df_out

# Prepare lagged remainder data for ARMAX
def prepare_deseasonal_armax_data(df, y_col, exog_cols, forecast_horizon=3, lags=1, train_split=0.8):
    dfc = df.copy()
    for i in range(1, lags+1):
        dfc[f'remainder_lag_{i}'] = dfc['deseasonalized_remainder'].shift(i)
    dfc['target'] = dfc['deseasonalized_remainder'].shift(-forecast_horizon)
    cols = ['target'] + exog_cols + [f'remainder_lag_{i}' for i in range(1, lags+1)] + ['seasonal_component','trend_component', y_col]
    dfm = dfc[cols].dropna()
    split = int(len(dfm) * train_split)
    train = dfm.iloc[:split]
    test = dfm.iloc[split:]
    exog_feats = exog_cols + [f'remainder_lag_{i}' for i in range(1, lags+1)]
    return train['target'], train[exog_feats], test['target'], test[exog_feats], test, dfm

# Fit ARMAX model
def train_armax_model(endog, exog, ar_order=1, ma_order=0):
    model = SARIMAX(endog, exog=exog, order=(ar_order,0,ma_order),
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

# Forecast using fitted model
def forecast_armax_model(model_fit, steps, exog):
    return model_fit.forecast(steps=steps, exog=exog)

# Evaluate by reconstructing full price
def evaluate_deseasonalized(test_df, forecast_rem, y_col, plot=False):
    forecast_full = forecast_rem + test_df['seasonal_component'] + test_df['trend_component']
    actual = test_df[y_col]
    rmse = np.sqrt(mean_squared_error(actual, forecast_full))
    if plot:
        # Downsample to reduce clutter
        step = max(1, len(actual) // 1000)
        x = actual.index[::step]
        y_actual = actual[::step]
        y_forecast = forecast_full[::step]

        plt.figure(figsize=(14, 6))
        plt.plot(x, y_actual, label='Actual', linewidth=1, alpha=1)
        plt.plot(x, y_forecast, label='ARMAX (p=0, q=0, b=2)', linewidth=1, alpha=1)

        plt.title("Actual ID1 vs 24h ahead ID1 Forecast", fontsize=16)
        plt.xlabel("Date", fontsize=16)
        plt.ylabel("EUR per MW", fontsize=16)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    return rmse, forecast_full

# Generate forecasts and save to CSV
def generate_and_save_forecasts(full_data, forecast_values, y_col, output_file):
    # Create a common index by getting the intersection of both indices
    common_index = full_data.index.intersection(forecast_values.index)

    # Create output dataframe with aligned indices
    output_df = pd.DataFrame({
        'timestamp': common_index,
        'actual_' + y_col: full_data.loc[common_index],
        'forecast_' + y_col: forecast_values.loc[common_index]
    })

    # Reset index to make timestamp a column
    output_df.reset_index(drop=True, inplace=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Forecasts saved to {output_file}")
    print(f"Saved {len(output_df)} records to CSV file.")
    return output_df

# Grid search for optimal ARMAX parameters
def grid_search_armax(df, y_col, exog_cols, forecast_horizon, train_split, grid_params):
    """
    Perform grid search to find optimal ARMAX parameters.

    Args:
        df: Deseasonalized dataframe
        y_col: Target column name
        exog_cols: List of exogenous variable column names
        forecast_horizon: Forecast horizon in hours
        train_split: Train-test split ratio
        grid_params: Dict with parameter grid for lags, ar_order, and ma_order

    Returns:
        Tuple of (best_params, results_df)
    """
    print("\n" + "="*50)
    print("Starting grid search for optimal ARMAX parameters...")
    print("="*50)

    # Generate parameter combinations
    lags_list = grid_params['lags']
    ar_orders = grid_params['ar_order']
    ma_orders = grid_params['ma_order']

    # For storing results
    results = []

    # Best parameters tracking
    best_rmse = float('inf')
    best_params = {}
    total_combinations = len(lags_list) * len(ar_orders) * len(ma_orders)

    print(f"Grid search will evaluate {total_combinations} parameter combinations")
    print(f"Parameters: lags {lags_list}, AR orders {ar_orders}, MA orders {ma_orders}")
    print("-"*50)

    counter = 0

    # Iterate through parameter combinations
    for lags in lags_list:
        # Skip the case where lags=0 as it would create no lag features
        if lags == 0:
            # For lags=0, we only need to try models without AR terms
            temp_ar_orders = [0]
        else:
            temp_ar_orders = ar_orders

        for ar in temp_ar_orders:
            for ma in ma_orders:
                counter += 1
                print(f"\nTrying combination {counter}/{total_combinations}: lags={lags}, ar_order={ar}, ma_order={ma}")

                try:
                    # Prepare data with current lag value
                    y_train, X_train, y_test, X_test, test_df, full_df = prepare_deseasonal_armax_data(
                        df, y_col, exog_cols, forecast_horizon=forecast_horizon,
                        lags=lags, train_split=train_split
                    )

                    # Train model with current AR and MA orders
                    fit = train_armax_model(y_train, X_train, ar_order=ar, ma_order=ma)

                    # Generate out-of-sample forecasts
                    preds_rem_test = forecast_armax_model(fit, steps=len(y_test), exog=X_test)

                    # Calculate RMSE
                    out_of_sample_rmse, _ = evaluate_deseasonalized(test_df, preds_rem_test, y_col, plot=False)

                    # Get AIC and BIC metrics
                    aic = fit.aic
                    bic = fit.bic

                    # Update best parameters if current RMSE is better
                    if out_of_sample_rmse < best_rmse:
                        best_rmse = out_of_sample_rmse
                        best_params = {'lags': lags, 'ar_order': ar, 'ma_order': ma}
                        print(f"New best model found! RMSE: {best_rmse:.2f} €/MWh")

                    # Store results
                    results.append({
                        'lags': lags,
                        'ar_order': ar,
                        'ma_order': ma,
                        'out_of_sample_rmse': out_of_sample_rmse,
                        'aic': aic,
                        'bic': bic
                    })

                    print(f"Completed: lags={lags}, ar_order={ar}, ma_order={ma}, RMSE={out_of_sample_rmse:.2f}, AIC={aic:.2f}, BIC={bic:.2f}")

                except Exception as e:
                    print(f"Error with lags={lags}, ar_order={ar}, ma_order={ma}: {str(e)}")
                    results.append({
                        'lags': lags,
                        'ar_order': ar,
                        'ma_order': ma,
                        'out_of_sample_rmse': None,
                        'aic': None,
                        'bic': None,
                        'error': str(e)
                    })

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Sort by RMSE
    results_df = results_df.sort_values('out_of_sample_rmse')

    print("\n" + "="*50)
    print("Grid search completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best out-of-sample RMSE: {best_rmse:.2f} €/MWh")
    print("="*50)
    print("\nTop 5 parameter combinations:")
    print(results_df.head(5).to_string(index=False))
    print("="*50)

    return best_params, results_df

# Main function to execute the forecasting pipeline
def run_forecast():
    # Get parameters from config
    file_path = CONFIG['input_file']
    y_col = CONFIG['target_col']
    date_col = CONFIG['date_col']
    exog_cols = CONFIG['exog_cols']
    forecast_horizon = CONFIG['forecast_horizon']
    train_split = CONFIG['train_split']
    plot = CONFIG['plot_results']
    output_file = CONFIG['output_file']
    grid_search = CONFIG.get('grid_search', False)

    # Load and preprocess data
    print(f"Loading data from {file_path}...")
    df0 = preprocess_data(file_path)

    # Deseasonalize data
    print("Deseasonalizing data...")
    df = deseasonalize_series(df0, date_col, y_col, plot=plot)

    # Check if grid search is enabled
    if grid_search:
        grid_params = CONFIG['grid_params']
        best_params, results_df = grid_search_armax(
            df, y_col, exog_cols, forecast_horizon, train_split, grid_params
        )

        # Save grid search results
        results_file = output_file.replace('.csv', '_grid_search_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Grid search results saved to {results_file}")

        # Use best parameters for final model
        lags = best_params['lags']
        ar_order = best_params['ar_order']
        ma_order = best_params['ma_order']

        print(f"\nUsing optimal parameters for final model: lags={lags}, ar_order={ar_order}, ma_order={ma_order}")
    else:
        # Use parameters from CONFIG if not doing grid search
        lags = CONFIG['lags']
        ar_order = CONFIG['ar_order']
        ma_order = CONFIG['ma_order']

    # Prepare data for ARMAX model with optimal or configured parameters
    print("\nPreparing data for final ARMAX model...")
    y_train, X_train, y_test, X_test, test_df, full_df = prepare_deseasonal_armax_data(
        df, y_col, exog_cols, forecast_horizon=forecast_horizon,
        lags=lags, train_split=train_split
    )

    # Train ARMAX model
    print("Training final ARMAX model...")
    fit = train_armax_model(y_train, X_train, ar_order=ar_order, ma_order=ma_order)

    # Generate forecasts for test set (out-of-sample)
    print(f"Generating forecasts for {len(y_test)} out-of-sample time points...")
    preds_rem_test = forecast_armax_model(fit, steps=len(y_test), exog=X_test)

    # Evaluate model performance on test set
    print("Evaluating model on test set...")
    out_of_sample_rmse, forecasts_test = evaluate_deseasonalized(test_df, preds_rem_test, y_col, plot=plot)
    print(f"Out-of-sample RMSE: {out_of_sample_rmse:.2f} €/MWh")

    # Generate forecasts for training set (in-sample)
    print(f"Generating forecasts for {len(y_train)} in-sample time points...")
    preds_rem_train = fit.get_prediction(start=0, end=len(y_train)-1).predicted_mean

    # Get training data frame for reconstruction
    train_df = full_df.iloc[:int(len(full_df) * train_split)]

    # Reconstruct in-sample forecasts
    forecasts_train = preds_rem_train + train_df['seasonal_component'] + train_df['trend_component']

    # Calculate in-sample RMSE
    in_sample_rmse = np.sqrt(mean_squared_error(train_df[y_col], forecasts_train))
    print(f"In-sample RMSE: {in_sample_rmse:.2f} €/MWh")

    # Create a complete forecast series for both in-sample and out-of-sample
    print("Combining in-sample and out-of-sample forecasts...")
    all_forecasts = pd.Series(index=full_df.index, dtype=float)
    split_idx = int(len(full_df) * train_split)

    # Assign in-sample forecasts
    all_forecasts.iloc[:split_idx] = forecasts_train.values

    # Assign out-of-sample forecasts
    all_forecasts.iloc[split_idx:] = forecasts_test.values

    # Calculate overall RMSE
    overall_rmse = np.sqrt(mean_squared_error(
        full_df[y_col],
        all_forecasts[~all_forecasts.isna()]
    ))
    print(f"Overall RMSE (in-sample and out-of-sample): {overall_rmse:.2f} €/MWh")

    # Print summary of RMSE metrics
    print("\nRMSE Summary:")
    print(f"{'Metric':<25} {'Value (€/MWh)':<15}")
    print("-" * 40)
    print(f"{'In-sample RMSE':<25} {in_sample_rmse:.2f}")
    print(f"{'Out-of-sample RMSE':<25} {out_of_sample_rmse:.2f}")
    print(f"{'Overall RMSE':<25} {overall_rmse:.2f}")

    # Save forecasts to CSV
    print("\nSaving forecasts to CSV...")
    output_df = generate_and_save_forecasts(df0[y_col], all_forecasts, y_col, output_file)

    return output_df, in_sample_rmse, out_of_sample_rmse, overall_rmse

if __name__ == '__main__':
    run_forecast()
