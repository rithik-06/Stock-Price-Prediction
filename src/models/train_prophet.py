
## Train Prophet model for stock price prediction.

import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
import numpy as np


def train_prophet_model(ticker, data_file, mlflow_tracking_uri):
    """
    Train Prophet model and log to MLflow.

    Args:
        ticker: Stock symbol
        data_file: Path to processed CSV with features
        mlflow_tracking_uri: MLflow tracking server URI
    """
    print("Training Prophet model for " + ticker)

    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("stock_price_prediction")

    # Read data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df['Close']
    })

    # Train/test split (80/20)
    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:split_idx]
    test_df = prophet_df[split_idx:]

    print("Train size: " + str(len(train_df)))
    print("Test size: " + str(len(test_df)))

    # Start MLflow run
    with mlflow.start_run(run_name="Prophet_" + ticker):
        # Log parameters
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))

        # Train model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )

        model.fit(train_df)

        # Make predictions on test set
        forecast = model.predict(test_df[['ds']])

        # Calculate metrics
        predictions = forecast['yhat'].values
        actual = test_df['y'].values

        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        print("MAE: " + str(mae))
        print("RMSE: " + str(rmse))
        print("MAPE: " + str(mape) + "%")

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mape", mape)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Model logged to MLflow successfully!")

        return model, mae, rmse, mape


if __name__ == "__main__":
    # Test with AAPL
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "processed" / "AAPL_features.csv"
    mlflow_uri = "file://" + str(project_root / "mlruns")

    if data_file.exists():
        train_prophet_model("AAPL", str(data_file), mlflow_uri)
    else:
        print("Data file not found: " + str(data_file))