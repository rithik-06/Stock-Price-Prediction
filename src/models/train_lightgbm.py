
#Train LightGBM model for stock price prediction.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.lightgbm
from pathlib import Path


def prepare_data(df, target_col='Close'):
    """Prepare features for LightGBM."""
    df = df.dropna()
    df['Target'] = df[target_col].shift(-1)
    df = df[:-1]

    feature_cols = [col for col in df.columns if col not in
                    ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    X = df[feature_cols]
    y = df['Target']

    return X, y, feature_cols


def train_lightgbm_model(ticker, data_file, mlflow_tracking_uri):
    """
    Train LightGBM model and log to MLflow.
    """
    print("Training LightGBM model for " + ticker)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("stock_price_prediction")

    # Read data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print("Original data shape: " + str(df.shape))

    # Prepare features
    X, y, feature_cols = prepare_data(df)
    print("Features: " + str(len(feature_cols)))
    print("Samples: " + str(len(X)))

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print("Train size: " + str(len(X_train)))
    print("Test size: " + str(len(X_test)))

    # Start MLflow run
    with mlflow.start_run(run_name="LightGBM_" + ticker):

        # Model parameters
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }

        # Log parameters
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))

        for key, value in params.items():
            if key != 'verbose':
                mlflow.log_param(key, value)

        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

        print("\nTraining Metrics:")
        print("MAE: " + str(train_mae))
        print("RMSE: " + str(train_rmse))
        print("MAPE: " + str(train_mape) + "%")

        print("\nTest Metrics:")
        print("MAE: " + str(test_mae))
        print("RMSE: " + str(test_rmse))
        print("MAPE: " + str(test_mape) + "%")

        # Log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("test_mape", test_mape)

        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_10 = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        print("\nTop 10 Important Features:")
        for feat, importance in top_10:
            print(feat + ": " + str(importance))
            mlflow.log_metric("feat_" + feat, importance)

        print("\nModel logged to MLflow successfully!")

        return model, test_mae, test_rmse, test_mape


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "processed" / "AAPL_features.csv"
    mlflow_uri = "file://" + str(project_root / "mlruns")

    if data_file.exists():
        train_lightgbm_model("AAPL", str(data_file), mlflow_uri)
    else:
        print("Data file not found: " + str(data_file))