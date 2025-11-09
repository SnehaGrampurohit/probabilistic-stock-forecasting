
# src/tech_indicators.py

import pandas as pd

try:
    import pandas_ta as ta
except Exception as e:
    raise ImportError(
        "pandas_ta is required for technical indicators on Streamlit Cloud. "
        "Add `pandas_ta` to requirements.txt and reinstall. Original error: "
        + str(e)
    )


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators using pandas_ta and append them to the DataFrame.

    Required input columns: ['Open','High','Low','Close','Volume']
    Output columns added (matching your app expectations):
      rsi, ma50, ma200, macd, signal_line, wpr, slowk, slowd, ult_oscillator,
      lag1, lag2, lag3
    """
    df = df.copy()

    # --- Safety checks
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for indicators: {sorted(missing)}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- Momentum / Oscillators
    df["rsi"] = ta.rsi(close, length=14)

    # Williams %R
    df["wpr"] = ta.willr(high, low, close, length=14)

    # Stochastic Oscillator (Slow %K / %D)
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    # pandas_ta returns columns like 'STOCHk_14_3_3' and 'STOCHd_14_3_3'
    if stoch is not None and not stoch.empty:
        df["slowk"] = stoch.iloc[:, 0]
        df["slowd"] = stoch.iloc[:, 1]
    else:
        df["slowk"] = pd.NA
        df["slowd"] = pd.NA

    # MACD (12,26,9)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    # Columns: 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    if macd is not None and not macd.empty:
        df["macd"] = macd["MACD_12_26_9"]
        df["signal_line"] = macd["MACDs_12_26_9"]
    else:
        df["macd"] = pd.NA
        df["signal_line"] = pd.NA

    # Moving Averages
    df["ma50"] = ta.sma(close, length=50)
    df["ma200"] = ta.sma(close, length=200)

    # Ultimate Oscillator (7,14,28)
    df["ult_oscillator"] = ta.uo(high, low, close, fast=7, medium=14, slow=28)

    # Lags of close
    df["lag1"] = df["Close"].shift(1)
    df["lag2"] = df["Close"].shift(2)
    df["lag3"] = df["Close"].shift(3)

    # Drop initial rows with NaNs caused by indicator windows
    df.dropna(inplace=True)

    return df
