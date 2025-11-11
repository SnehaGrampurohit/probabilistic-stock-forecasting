# src/train_offline.py
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tech_indicators import calculate_technical_indicators

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AMZN"
MONTHS = 60          # training window
N_FEATURES = 7       # features kept by RFE
TEST_DAYS = 7        # last 7 days as test set

ALL_FEATURES = [
    'Open','High','Low','Volume',
    'wpr','rsi','slowk','slowd','macd','ma50','ma200','ult_oscillator',
    'day','month','quarter','year','lag1','lag2','lag3'
]

def fetch_data(symbol=SYMBOL, months=MONTHS):
    end = datetime.today()
    start = pd.Timestamp(end) - pd.DateOffset(months=months)
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No data from yfinance.")
    df.columns = df.columns.get_level_values(0)
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df = calculate_technical_indicators(df)
    df['Date'] = df.index
    return df

def temporal_split(df, test_days=TEST_DAYS):
    train_df = df.iloc[:-test_days]
    test_df  = df.iloc[-test_days:]
    return train_df, test_df

def train_rf_rfe(train_df):
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df['Close'].values

    base_rf = RandomForestRegressor(
        n_estimators=300, max_depth=30, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rfe = RFE(estimator=base_rf, n_features_to_select=N_FEATURES)
    X_train_sel = rfe.fit_transform(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=30, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sel, y_train)
    return rf, rfe

def evaluate(rf, rfe, test_df):
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df['Close'].values
    y_pred = rf.predict(rfe.transform(X_test))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"mae": mae, "rmse": rmse, "test_days": len(test_df)}

def main():
    print("Fetching data…")
    df = fetch_data()
    print(f"Rows: {len(df)}")

    print("Splitting train/test…")
    train_df, test_df = temporal_split(df)

    print("Training RF + RFE…")
    rf, rfe = train_rf_rfe(train_df)

    print("Evaluating…")
    metrics = evaluate(rf, rfe, test_df)
    print("Metrics:", metrics)

    print("Saving artifacts…")
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(rfe, MODELS_DIR / "rfe.pkl")
    (MODELS_DIR / "feature_list.json").write_text(json.dumps(ALL_FEATURES, indent=2))
    (MODELS_DIR / "model_card.json").write_text(json.dumps({
        "model": "RandomForestRegressor + RFE",
        "symbol": SYMBOL,
        "trained_months": MONTHS,
        "n_features_selected": N_FEATURES,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics
    }, indent=2))
    print("Saved to:", MODELS_DIR)

if __name__ == "__main__":
    main()