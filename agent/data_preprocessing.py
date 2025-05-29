import pandas as pd

def load_and_preprocess_data(file_path):
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
    df = df[df['Date'].dt.year == 2023]

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # Z-score the ID1_price column
    df['ID1_price_z_score'] = (df['ID1_price'] - df['ID1_price'].mean()) / df['ID1_price'].std()

    # Create rolling z-score with a window of 24 hours
    rolling_mean = df['ID1_price'].rolling(window=24, min_periods=1).mean()
    rolling_std = df['ID1_price'].rolling(window=24, min_periods=1).std()
    # Use 1 as the std where it's 0 to avoid division by zero
    rolling_std = rolling_std.replace(0, 1)
    df['ID1_price_rolling_z_score'] = (df['ID1_price'] - rolling_mean) / rolling_std

    # Fill NaN values in rolling z-score with 0
    df['ID1_price_rolling_z_score'].fillna(0, inplace=True)

    # Add column with 1 if price is positive, else 0
    df['ID1_price_positive'] = (df['ID1_price'] > 0).astype(int)







    # print min  and max of ID1_price
    print(f"[DATA PREPROCESSING] Min ID1_price: {df['ID1_price'].min()}, Max ID1_price: {df['ID1_price'].max()}")



    df.to_csv("../data/preprocessed_data_2023.csv", index=False)

    return df

# Example usage:
df = load_and_preprocess_data("../data/ID1_prices_germany.csv")
