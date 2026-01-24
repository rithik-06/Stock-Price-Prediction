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

}
