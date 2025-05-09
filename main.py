import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_FILE_PATH = 'dataset/dutch-weather/input.csv'
DATE_COLUMN = 'YYYYMMDD'
VALUE_COLUMN = 'TX'
PERIODS_TO_FORECAST = 24 # Number of periods to forecast
FORECAST_FREQ = 'M' # Frequency of the forecast ('D' for day, 'M' for month, 'H' for hour, etc.)

# --- Load and Prepare Data ---
try:
    # Load the dataset
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Convert '20060701' to correct date format
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='%Y%m%d')
    
    # Value is in 0.1 degrees Celsius convert
    df[VALUE_COLUMN] = df[VALUE_COLUMN] / 10.0

    # Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
    df_prophet = df.rename(columns={DATE_COLUMN: 'ds', VALUE_COLUMN: 'y'})

    # Keep only 'ds' and 'y' columns
    df_prophet = df_prophet[['ds', 'y']]

    print("Data loaded and prepared successfully.")
    print(df_prophet.head())

except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found in the CSV. Please check DATE_COLUMN and VALUE_COLUMN.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Initialize and Fit Prophet Model ---
# Initialize the Prophet model
# You can add seasonality, holidays, etc. here if needed
# For example: model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

# Fit the model to your data
try:
    model.fit(df_prophet)
    print("Prophet model fitted successfully.")
except Exception as e:
    print(f"An error occurred during model fitting: {e}")
    exit()

# --- Make Future Predictions ---
# Create a dataframe for future dates
try:
    future = model.make_future_dataframe(periods=PERIODS_TO_FORECAST, freq=FORECAST_FREQ)
    print(f"\nFuture dataframe created for {PERIODS_TO_FORECAST} {FORECAST_FREQ}:")
    print(future.tail())

    # Make predictions
    forecast = model.predict(future)
    print("\nForecast generated successfully.")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
except Exception as e:
    print(f"An error occurred during prediction: {e}")
    exit()

# --- Plot the Forecast ---
try:
    print("\nPlotting forecast...")
    # Plot the forecast
    fig1 = model.plot(forecast)
    plt.title(f'Prophet Forecast for {VALUE_COLUMN}')
    plt.xlabel('Date')
    plt.ylabel(VALUE_COLUMN)
    plt.show()

    # Plot forecast components (trend, seasonality)
    fig2 = model.plot_components(forecast)
    plt.show()
    print("Plots displayed. Close plot windows to continue.")
except Exception as e:
    print(f"An error occurred during plotting: {e}")

print("\nScript finished.")