import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/preprocessed_data_2024.csv")

def remove_outliers(df, date_col, y_col, threshold=3, plot=False):
    """
    Plot the time series data and highlight outliers based on a z-score threshold.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    date_col (str): The column name for the date.
    y_col (str): The column name for the values to check for outliers.
    threshold (float): Z-score threshold to identify outliers.
    """
    df['z_score'] = (df[y_col] - df[y_col].mean()) / df[y_col].std()

    outliers = df[df['z_score'].abs() > threshold]

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[y_col], label='Data')
        plt.scatter(outliers[date_col], outliers[y_col], color='red', label='Outliers', marker='o')
        plt.xticks([])  # Remove x-axis ticks
        plt.ylabel(y_col)
        plt.title('Time Series Data with Outliers Highlighted')
        plt.legend()
        plt.show()



    # Replace outliers with the value 24h before
    df[y_col] = df[y_col].where(df['z_score'].abs() <= threshold, df[y_col].shift(24))

    df.drop(columns=['z_score'], inplace=True)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[y_col], label='Data without Outliers')
        plt.xticks([])  # Remove x-axis ticks
        plt.ylabel(y_col)
        plt.title('Time Series Data After Outlier Replacement')
        plt.legend()
        plt.show()

    return df

# Example usage
#remove_outliers(df, 'Date', 'ID1_price', threshold=10, plot=True)

