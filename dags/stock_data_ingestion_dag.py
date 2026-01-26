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


# Task function to fetch multiple stocks
def fetch_stock_data_task():
    """Fetch multiple stock tickers and save to CSV."""

    # List of stocks to fetch
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

    for ticker in tickers:
        print("Starting to fetch " + ticker + " stock data...")

        try:
            # Download data
            df = yf.download(ticker, period='5y', progress=False)

            if df.empty:
                print("No data found for " + ticker)
                continue

            print("Fetched " + str(len(df)) + " records for " + ticker)

            # Save to CSV
            output_dir = Path('/opt/airflow/data/raw')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d")
            filename = output_dir / (ticker + "_" + timestamp + ".csv")

            df.to_csv(filename)
            print("Saved " + ticker + " to: " + str(filename))

        except Exception as e:
            print("Error fetching " + ticker + ": " + str(e))
            continue

    print("Finished fetching all stocks!")

# Create the task
fetch_stocks_task = PythonOperator(
    task_id='fetch_multiple_stocks',
    python_callable=fetch_stock_data_task,
    dag=dag,
)


# Task sequence
fetch_stocks_task