from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def deseasonalize_data(df, date_col, y_col, plot=False):
    from prophet import Prophet
    import pandas as pd
    import matplotlib.pyplot as plt

    # Safely copy and rename
    df_prophet = df[[date_col, y_col]].copy()
    df_prophet.rename(columns={date_col: 'ds', y_col: 'y'}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Fit Prophet model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)

    # Predict components
    forecast = model.predict(df_prophet)
    df_prophet['seasonal_component'] = forecast['daily'] + forecast['weekly']
    df_prophet['trend_component'] = forecast['trend']
    df_prophet['deseasonalized_remainder'] = df_prophet['y'] - df_prophet['seasonal_component'] - df_prophet['trend_component']

    if plot:
        model.plot_components(forecast)
        plt.show()

    # Restore original column names
    df_prophet.rename(columns={'ds': date_col, 'y': y_col}, inplace=True)

    df_out = df.merge(
        df_prophet[[date_col, 'seasonal_component', 'trend_component', 'deseasonalized_remainder']],
        on=date_col,
        how='left'
    )

    return df_out









