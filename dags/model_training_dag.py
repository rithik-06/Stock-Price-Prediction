"""
Airflow DAG to generate features from raw stock data.
Runs after stock data ingestion.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Prophet
from prophet import Prophet

# XGBoost
import xgboost as xgb

# LightGBM
import lightgbm as lgb

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MLflow
import mlflow
import mlflow.sklearn

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
    'model_training',
    default_args=default_args,
    description='Train ML models on all stocks weekly',
    schedule_interval='0 19 * * 0',  # Every Sunday at 7 PM
    catchup=False,
    tags=['stock', 'ml', 'training'],
)


# Helper function to prepare data
def prepare_data_for_ml(df, target_col='Close'):
    """Prepare features for ML models."""
    df = df.dropna()
    df['Target'] = df[target_col].shift(-1)
    df = df[:-1]

    feature_cols = [col for col in df.columns if col not in
                    ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    X = df[feature_cols]
    y = df['Target']

    return X, y, feature_cols


# Task 1: Train Prophet models
def train_prophet_models():
    """Train Prophet model for all stocks."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("stock_price_prediction")

    for ticker in tickers:
        print("Training Prophet for " + ticker)

        try:
            # Load data
            data_file = "/opt/airflow/data/processed/" + ticker + "_features.csv"
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)

            # Prepare for Prophet
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close']
            })

            split_idx = int(len(prophet_df) * 0.8)
            train_df = prophet_df[:split_idx]
            test_df = prophet_df[split_idx:]

            # Start MLflow run
            with mlflow.start_run(run_name="Prophet_" + ticker):
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", "Prophet")
                mlflow.log_param("train_size", len(train_df))
                mlflow.log_param("test_size", len(test_df))

                # Train
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.05
                )
                model.fit(train_df)

                # Predict
                forecast = model.predict(test_df[['ds']])
                predictions = forecast['yhat'].values
                actual = test_df['y'].values

                # Metrics
                mae = np.mean(np.abs(predictions - actual))
                rmse = np.sqrt(np.mean((predictions - actual) ** 2))
                mape = np.mean(np.abs((actual - predictions) / actual)) * 100

                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mape", mape)

                print(ticker + " Prophet - MAE: " + str(mae) + ", MAPE: " + str(mape) + "%")

        except Exception as e:
            print("Error training Prophet for " + ticker + ": " + str(e))
            continue

    print("Prophet training complete!")


# Task 2: Train XGBoost models
def train_xgboost_models():
    """Train XGBoost model for all stocks."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("stock_price_prediction")

    for ticker in tickers:
        print("Training XGBoost for " + ticker)

        try:
            data_file = "/opt/airflow/data/processed/" + ticker + "_features.csv"
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)

            X, y, feature_cols = prepare_data_for_ml(df)

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            with mlflow.start_run(run_name="XGBoost_" + ticker):
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", "XGBoost")
                mlflow.log_param("n_features", len(feature_cols))

                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'random_state': 42
                }

                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mape", mape)

                print(ticker + " XGBoost - MAE: " + str(mae) + ", MAPE: " + str(mape) + "%")

        except Exception as e:
            print("Error training XGBoost for " + ticker + ": " + str(e))
            continue

    print("XGBoost training complete!")


# Task 3: Train LightGBM models
def train_lightgbm_models():
    """Train LightGBM model for all stocks."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")
    mlflow.set_experiment("stock_price_prediction")

    for ticker in tickers:
        print("Training LightGBM for " + ticker)

        try:
            data_file = "/opt/airflow/data/processed/" + ticker + "_features.csv"
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)

            X, y, feature_cols = prepare_data_for_ml(df)

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            with mlflow.start_run(run_name="LightGBM_" + ticker):
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", "LightGBM")
                mlflow.log_param("n_features", len(feature_cols))

                params = {
                    'objective': 'regression',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'n_estimators': 100,
                    'random_state': 42,
                    'verbose': -1
                }

                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mape", mape)

                print(ticker + " LightGBM - MAE: " + str(mae) + ", MAPE: " + str(mape) + "%")

        except Exception as e:
            print("Error training LightGBM for " + ticker + ": " + str(e))
            continue

    print("LightGBM training complete!")


# Create tasks
train_prophet_task = PythonOperator(
    task_id='train_prophet_models',
    python_callable=train_prophet_models,
    dag=dag,
)

train_xgboost_task = PythonOperator(
    task_id='train_xgboost_models',
    python_callable=train_xgboost_models,
    dag=dag,
)

train_lightgbm_task = PythonOperator(
    task_id='train_lightgbm_models',
    python_callable=train_lightgbm_models,
    dag=dag,
)

# All three can run in parallel
[train_prophet_task, train_xgboost_task, train_lightgbm_task]