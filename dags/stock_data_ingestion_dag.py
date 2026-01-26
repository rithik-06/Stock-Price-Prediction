"""
Airflow DAG to fetch stock data daily.
Runs every day at 6 PM to download stock prices.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from pathlib import Path
import os

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'stock_data_ingestion',
    default_args=default_args,
    description='Fetch stock data from Yahoo Finance daily',
    schedule_interval='0 18 * * 1-5',
    catchup=False,
    tags=['stock', 'data-ingestion'],
)


# Task function to fetch stock data
def fetch_apple_stock():
    """Fetch AAPL stock data and save to CSV."""
    print("Starting to fetch AAPL stock data...")

    # Download data
    ticker = 'AAPL'
    df = yf.download(ticker, period='1mo', progress=False)

    if df.empty:
        raise Exception("No data found for AAPL")

    print("Fetched " + str(len(df)) + " records")

    # Save to CSV
    output_dir = Path('/opt/airflow/data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    filename = output_dir / (ticker + "_" + timestamp + ".csv")

    df.to_csv(filename)
    print("Saved to: " + str(filename))


# Create the task
fetch_aapl_task = PythonOperator(
    task_id='fetch_aapl_stock',
    python_callable=fetch_apple_stock,
    dag=dag,
)

# Task sequence
fetch_aapl_task