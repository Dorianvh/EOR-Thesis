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

    return df[['ID1_price', 'Hour', 'DayOfWeek', 'Month']]