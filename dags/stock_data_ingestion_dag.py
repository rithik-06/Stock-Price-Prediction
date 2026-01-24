## airflow dag to fetch stock data daily

## runs every day at 6 PM to download stck prices.

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import  os

from src.data.fetch_stock_data import project_root

## add project root to python path so we dan import our modules

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.fetch_stock_data import fetch_stock_data

## default arguments for the DAG

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
    schedule_interval='0 18 * * 1-5',  # Run at 6 PM on weekdays (Mon-Fri)
    catchup=False,
    tags=['stock', 'data-ingestion'],
)

# Task 1: Fetch Apple stock data
def fetch_apple_stock():
    """Fetch AAPL stock data and save to CSV."""
    data_dir = os.path.join(project_root, 'data', 'raw')
    print("Starting to fetch AAPL stock data...")
    df = fetch_stock_data('AAPL', period='1mo', output_dir=data_dir)
    if df is not None:
        print("Successfully fetched AAPL data")
    else:
        raise Exception("Failed to fetch AAPL data")


# Create the task
fetch_aapl_task = PythonOperator(
    task_id='fetch_aapl_stock',
    python_callable=fetch_apple_stock,
    dag=dag,
)

# Task sequence (just one task for now)
fetch_aapl_task