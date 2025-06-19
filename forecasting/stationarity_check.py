import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from forecasting.deseasonalize import deseasonalize_data

df = pd.read_csv("../data/preprocessed_data_2023.csv")
df['Date'] = pd.to_datetime(df['Date'])

#df = deseasonalize_data(df, 'Date', 'ID1_price')

# check for stationarity

def check_stationarity(df, column):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test and KPSS Test.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    column (str): The column name of the time series to check.

    Returns:
    bool: True if the series is stationary, False otherwise.
    """
    adf_result = sm.tsa.stattools.adfuller(df[column])
    kpss_result = sm.tsa.stattools.kpss(df[column], regression='c')

    adf_statistic, adf_p_value = adf_result[0], adf_result[1]
    kpss_statistic, kpss_p_value = kpss_result[0], kpss_result[1]

    print(f"ADF Statistic: {adf_statistic}, p-value: {adf_p_value}")
    print(f"KPSS Statistic: {kpss_statistic}, p-value: {kpss_p_value}")

    # ADF test: p-value < 0.05 indicates stationarity
    # KPSS test: p-value > 0.05 indicates stationarity
    return adf_p_value < 0.05 and kpss_p_value > 0.05

is_stationary = check_stationarity(df, 'ID1_price')
if is_stationary:
    print("The time series is stationary.")
else:
    print("The time series is not stationary. Consider differencing or transformation.")

# Plot the deseasonalized, and original data
#plt.figure(figsize=(12, 6))
#plt.plot(df['Date'], df['ID1_price'], label='Original Data', color='blue')
#plt.plot(df['Date'], df['deseasonalized_remainder'], label='Deseasonalized Data', color='orange')

plt.ylabel('Price')
plt.title('Original vs Deseasonalized Data')
plt.legend()
plt.show()
