from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def deseasonalize_data(df, date_col, y_col, plot = False):
    df_prophet = df.rename(columns={date_col: 'ds', y_col: 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(df_prophet)

    # Make forecast on the same data
    forecast = model.predict(df_prophet)

    # Compute total seasonal component
    df_prophet['seasonal_component'] = forecast['daily'] + forecast['weekly']
    df_prophet['trend_component'] = forecast['trend']
    # Merge deseasonalized data into the original DataFrame
    df_prophet['deseasonalized_remainder'] = df_prophet['y'] - df_prophet['seasonal_component']

    if plot:
        model.plot_components(forecast)
        plt.show()

    # Rename columns back to original names
    df_prophet = df_prophet.rename(columns={'ds': date_col,'y': y_col})

    return df_prophet







