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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from tech_indicators import calculate_technical_indicators

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AMZN"
MONTHS = 60          # training window length
N_FEATURES = 5       # features kept by RFE
TEST_DAYS = 7        # last 7 days as test set

ALL_FEATURES = [
    "Open", "High", "Low", "Volume",
    "wpr", "rsi", "slowk", "slowd", "macd", "ma50", "ma200", "ult_oscillator",
    "day", "month", "quarter", "year", "lag1", "lag2", "lag3",
]

# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def fetch_data(symbol: str = SYMBOL, months: int = MONTHS) -> pd.DataFrame:
    end = datetime.today()
    start = pd.Timestamp(end) - pd.DateOffset(months=months)

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data from yfinance.")

    # flatten multi-index if present
    df.columns = df.columns.get_level_values(0)

    # calendar features
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    # technical indicators + lags
    df = calculate_technical_indicators(df)
    df["Date"] = df.index

    return df


def temporal_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    return train_df, test_df

# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------
def train_rf_rfe(train_df: pd.DataFrame):
    """
    1) Run RFE with a baseline RF to select N_FEATURES.
    2) On the selected features, run GridSearchCV (TimeSeriesSplit)
       to find good RF hyperparameters.
    """
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df["Close"].values

    # ---- Step 1: RFE for feature selection (baseline RF) ----
    base_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rfe = RFE(estimator=base_rf, n_features_to_select=N_FEATURES)
    X_train_sel = rfe.fit_transform(X_train, y_train)

    # Which features were kept?
    selected_mask = rfe.support_
    selected_features = [f for f, keep in zip(ALL_FEATURES, selected_mask) if keep]
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    # ---- Step 2: Hyperparameter search on selected features ----
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
    }

    # TimeSeriesSplit respects temporal ordering
    cv = TimeSeriesSplit(n_splits=5)

    rf_for_search = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=rf_for_search,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train_sel, y_train)

    best_rf = grid.best_estimator_
    best_params = grid.best_params_
    print("Best RF params from GridSearchCV:", best_params)

    return best_rf, rfe, best_params


def evaluate(rf, rfe, test_df: pd.DataFrame):
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df["Close"].values
    X_test_sel = rfe.transform(X_test)

    y_pred = rf.predict(X_test_sel)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"mae": mae, "rmse": rmse, "test_days": len(test_df)}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Fetching data…")
    df = fetch_data()
    print(f"Rows: {len(df)}")

    print("Splitting train/test…")
    train_df, test_df = temporal_split(df)

    print("Training RF + RFE with hyperparameter search…")
    rf, rfe, best_params = train_rf_rfe(train_df)

    print("Evaluating on hold-out test window…")
    metrics = evaluate(rf, rfe, test_df)
    print("Metrics:", metrics)

    print("Saving artifacts…")
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(rfe, MODELS_DIR / "rfe.pkl")
    (MODELS_DIR / "feature_list.json").write_text(json.dumps(ALL_FEATURES, indent=2))

    model_card = {
        "model": "RandomForestRegressor + RFE",
        "symbol": SYMBOL,
        "trained_months": MONTHS,
        "n_features_selected": N_FEATURES,
        "selected_features": [
            f for f, keep in zip(ALL_FEATURES, rfe.support_) if keep
        ],
        "hyperparameters": best_params,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))
    print("Saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()