
import talib


def calculate_technical_indicators(df):
    """
    This function calculates technical indicators using the talib library and adds them to the DataFrame.
    """

    # Calculate technical indicators
    df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
    df['ma50'] = df['Close'].rolling(window=50).mean()
    df['ma200'] = df['Close'].rolling(window=200).mean()
    df['macd'], df['signal_line'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['wpr'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3,
                                           slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['ult_oscillator'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    #df['mfi'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)

    # Add lag features
    df['lag1'] = df['Close'].shift(1)  # Price from the previous day
    df['lag2'] = df['Close'].shift(2)  # Price from two days ago
    df['lag3'] = df['Close'].shift(3)  # Price from three days ago

    # Drop NaN values after adding indicators
    df.dropna(axis=0, inplace=True)

    return df
