import pandas as pd


def load_and_preprocess_data(file_path,year):
    """
    Load and preprocess data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with necessary columns.
    """
    print(f"[DATA PREPROCESSING] Loading data from {file_path}...")

    # Load the data
    df = pd.read_csv(file_path, parse_dates=["Date"])

    # Check for necessary columns
    if 'ID1_price' not in df.columns:
        raise ValueError("The input CSV must contain 'ID1_price' column.")

    # Drop rows with NaN values in 'ID1_price'
    df = df.dropna(subset=['ID1_price'])

   # Only keep data from 2023
    df = df[df['Date'].dt.year == year]

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    df.to_csv(f"../data/preprocessed_data_{year}.csv")

    return df

# scale specified columns in the DataFrame from -1 to 1 and roud to 4 decimal places
def scale_columns(df, columns,save=False):

    for column in columns:

        if column in df.columns:

            # forward fill NaN values
            df[column] = df[column].fillna(method='ffill')

            df[column] = df[column].round(2)
            col_name = column + "_scaled"
            min_val = df[column].min()
            max_val = df[column].max()
            df[col_name] = 2 * (df[column] - min_val) / (max_val - min_val) - 1
            df[col_name] = df[col_name].round(2)
        else:
            print(f"Column {column} not found in DataFrame.")


    if save:
        df.to_csv("../data/scaled_data_2023.csv", index=False)

    return df


# Example usage:
#df = load_and_preprocess_data("../data/ID1_prices_germany.csv",2023)

df = pd.read_csv("../data/preprocessed_data_2023.csv")
scaled_df = scale_columns(df, ['ID1_price', 'ARMAX_forecast_1hour', 'ARMAX_forecast_3hour', 'ARMAX_forecast_6hour'], True)


