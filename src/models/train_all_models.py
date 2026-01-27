
#### Train all models (Prophet, XGBoost, LightGBM) on all stocks.

from pathlib import Path
from train_prophet import train_prophet_model
from train_xgboost import train_xgboost_model
from train_lightgbm import train_lightgbm_model


def train_all():
    """Train all models on all stocks."""

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    project_root = Path(__file__).parent.parent.parent
    mlflow_uri = "file://" + str(project_root / "mlruns")

    results = []

    for ticker in tickers:
        print("\n" + "=" * 60)
        print("Processing " + ticker)
        print("=" * 60)

        data_file = project_root / "data" / "processed" / (ticker + "_features.csv")

        if not data_file.exists():
            print("Skipping " + ticker + " - no data file found")
            continue

        try:
            # Train Prophet
            print("\n--- Training Prophet ---")
            _, prophet_mae, prophet_rmse, prophet_mape = train_prophet_model(
                ticker, str(data_file), mlflow_uri
            )

            # Train XGBoost
            print("\n--- Training XGBoost ---")
            _, xgb_mae, xgb_rmse, xgb_mape = train_xgboost_model(
                ticker, str(data_file), mlflow_uri
            )

            # Train LightGBM
            print("\n--- Training LightGBM ---")
            _, lgb_mae, lgb_rmse, lgb_mape = train_lightgbm_model(
                ticker, str(data_file), mlflow_uri
            )

            results.append({
                'ticker': ticker,
                'prophet_mape': prophet_mape,
                'xgboost_mape': xgb_mape,
                'lightgbm_mape': lgb_mape
            })

        except Exception as e:
            print("Error training " + ticker + ": " + str(e))
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print("\nModel Performance by Stock (MAPE %):\n")
    print("Ticker  | Prophet | XGBoost | LightGBM | Best Model")
    print("-" * 60)

    for result in results:
        ticker = result['ticker']
        prophet = result['prophet_mape']
        xgb = result['xgboost_mape']
        lgb = result['lightgbm_mape']

        best = min(prophet, xgb, lgb)
        if best == prophet:
            best_model = "Prophet"
        elif best == xgb:
            best_model = "XGBoost"
        else:
            best_model = "LightGBM"

        print(ticker + "   | " + str(round(prophet, 2)) + "% | " +
              str(round(xgb, 2)) + "% | " + str(round(lgb, 2)) +
              "% | " + best_model)

    print("\nAll models trained! Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    train_all()