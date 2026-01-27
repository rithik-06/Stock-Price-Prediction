
##  Airflow DAG to generate features from raw stock data.
## uns after stock data ingestion.

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import os


# Feature calculation functions
def calculate_moving_averages(df):
    """Calculate Simple and Exponential Moving Averages."""
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    return df


def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df):
    """Calculate MACD and Signal line."""
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df


def calculate_volume_features(df):
    """Calculate volume-based features."""
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    return df


def calculate_price_features(df):
    """Calculate price-based features."""
    df['Returns'] = df['Close'].pct_change()
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_7'] = df['Close'].shift(7)
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = (df['HL_Range'] / df['Close']) * 100
    return df


def generate_features_for_stock(ticker, raw_dir, processed_dir):
    """Generate features for a single stock."""
    print("Processing features for " + ticker + "...")

    # Find latest raw file for this ticker
    raw_path = Path(raw_dir)
    files = list(raw_path.glob(ticker + "_*.csv"))

    if not files:
        print("No raw data found for " + ticker)
        return

    # Get most recent file
    input_file = max(files, key=os.path.getmtime)
    print("Reading from: " + str(input_file))

    # Read and clean data
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # Keep only OHLCV columns
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[columns_to_keep]

    # Convert to numeric
    for col in columns_to_keep:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    print("Original shape: " + str(df.shape))

    # Generate features
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_volume_features(df)
    df = calculate_price_features(df)

    # Drop NaN rows
    df = df.dropna()

    print("After features shape: " + str(df.shape))

    # Save
    output_file = Path(processed_dir) / (ticker + "_features.csv")
    df.to_csv(output_file)
    print("Saved to: " + str(output_file))


# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Create DAG

dag = DAG(
    'feature_engineering',
    default_args=default_args,
    description='Generate technical indicators from raw stock data',
    schedule_interval='30 18 * * 1-5',  # Run at 6:30 PM (after data ingestion)
    catchup=False,
    tags=['stock', 'features'],
)

# Task to generate features for all stocks
def generate_all_features():
    """Generate features for all stocks."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    raw_dir = '/opt/airflow/data/raw'
    processed_dir = '/opt/airflow/data/processed'

    for ticker in tickers:
        try:
            generate_features_for_stock(ticker, raw_dir, processed_dir)
        except Exception as e:
            print("Error processing " + ticker + ": " + str(e))
            continue

    print("Finished processing all stocks!")


# Create task
generate_features_task = PythonOperator(
    task_id='generate_features',
    python_callable=generate_all_features,
    dag=dag,
)

generate_features_task
