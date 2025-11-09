import streamlit as st
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

from tech_indicators import calculate_technical_indicators
from random_forest_model import run_random_forest
from gmm_model import run_gmm_model
from evaluation import evaluate_model
from plot_results import plot_results
from plot_results import plot_results_sub
from paper_methodology import paper_methodology
import pandas as pd



# Streamlit interface for number of years and feature selection
st.title("WELCOME TO PROBABILITY DISTRIBUTION PREDICTION OF STOCK PRICES DASHBOARD - AMAZON")

st.image('https://tse3.mm.bing.net/th/id/OIP.Cr9mnMQSaXgsyAJ0oQnQ6AHaD4?r=0&pid=Api', use_column_width=True)

# Tabs with icons
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Introduction",
    "🔍 Technical Indicators",
    "🧠 Model Prediction : Method 1",
    "🤖 Model Prediction: Method 2",
    # "📊 All Models Performance Comparison"
])

# Streamlit button for number of years
num_years = st.selectbox('Select number of years for fetching stock data:', [1, 2, 3])
num_features = st.selectbox('Select number of features for the model:', [5, 7, 9, 11, 16, 19])


# Fetch stock data based on user input
symbol = 'AMZN'
end_date = datetime(2024, 12, 31)
start_date = datetime(end_date.year - num_years, end_date.month, end_date.day)


@st.cache_data(ttl=3600)
def load_stock(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date, progress=False)

df = load_stock(symbol, start_date, end_date)

if df.empty:
    st.error(
        f"Yfinance Server down : ['AMZN']: YFRateLimitError('Too Many Requests. Rate limited. Try after a while."
    )
    st.stop()  # Do not proceed if df is empty

# Flatten the MultiIndex in the columns
df.columns = df.columns.get_level_values(0)



# Extract time components
df['day'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['year'] = df.index.year

# Call function to calculate technical indicators
df = calculate_technical_indicators(df)

# Split into train and test datasets
df['Date'] = df.index
test_end_date = df.index[-1]
test_start_date = test_end_date - pd.DateOffset(days=7)

test_df = df.tail(7)
train_df = df.iloc[:-7]


# Feature set
all_features = ['Open',
                'High',
                'Low',
                'Volume',
                'wpr',
                'rsi',
                'slowk',
                'slowd',
                'macd',
                'ma50',
                'ma200',
                'ult_oscillator',
                'day',
                'month',
                'quarter',
                'year',
                'lag1',
                'lag2',
                'lag3']
X_train = train_df[all_features].values
y_train = train_df['Close'].values
X_test = test_df[all_features].values
y_test = test_df['Close'].values

rf_model_base, rfe_base = run_random_forest(X_train, y_train, num_features)
X_train_selected = rfe_base.transform(X_train)
X_test_selected = rfe_base.transform(X_test)

# tab2 : Intro:
with (tab1):
    st.subheader("Introduction")
    st.write(
        "This application aims to predict stock closing prices by exploring two distinct methods for model development and optimization. We leverage historical stock data, including features like Open, High, Low, Volume, and RSI, to forecast future closing prices. The application implements two different methodologies that combine feature selection, machine learning models, and optimization techniques")

    st.subheader("Technical Indicators")
    st.write(
        "Technical indicators help to analyze price patterns, momentum, and market sentiment, aiding in better decision-making for trade entries and exits. They provide quantitative measures to identify trends, reversals, and potential price movements, increasing the accuracy of predictions.")

    st.write("The technical indicators used are RSI, WPR, MACD, MA50, MA200, Stochastic Oscillator")

    st.subheader(
        "Method 1: Probablistic Distribution of Stock Prices with Feature Selection using RFE, Random Forest, and Gaussian Mixture Model (GMM)")
    st.write(
        "In this approach, we start by applying Recursive Feature Elimination (RFE) to select the most relevant features for predicting stock closing prices. We then use a Random Forest Regressor for model training and prediction. To fine-tune the hyperparameters of the Random Forest, we employ Grid Search Cross-Validation (Grid Search CV), systematically exploring different parameter combinations to identify the best-performing model configuration.After obtaining predictions from each tree in the Random Forest, we further improve the model by incorporating a Gaussian Mixture Model (GMM). Specifically, we treat the predictions from each tree as samples for GMM fitting. Using the Bayesian Information Criterion (BIC), we determine the optimal number of components in the GMM to best capture the distribution of predictions. This hybrid approach ensures that we effectively model both feature importance and the underlying data distribution, enhancing prediction accuracy while balancing model complexity.")

    st.subheader("Relative Probability and Average Price")
    st.write(
        " The relative probability represents the model’s confidence in the most likely predicted price. Specifically, it measures how strongly the Gaussian Mixture Model (GMM) favors the most probable price, compared to all other plausible values. A higher value indicates that the model’s predictions are more concentrated around a single price, reflecting greater certainty.")

    st.write(
        "For example: If the relative probability is high, the model is confident that the closing price will be close to the predicted value. If it is low, the prediction is more uncertain or spread across several possible values.")

    st.write(
        "The average predicted price is the weighted average of all possible prices considered by the GMM, taking into account the probability of each scenario. This value represents the “expected” or “average” closing price according to the full probability distribution, providing a comprehensive summary of all likely outcomes.")

    st.write(
        "For example: Even if there are multiple possible price levels, the average predicted price gives a single summary value, balancing all scenarios by their likelihood.")

    st.subheader(
        "Method 2: Probablistic Distribution of Stock Prices using Gaussian Distribution and Optimized Weights using Maximum Likelihood Estimation (MLE)")
    st.write(
        "In the second approach, we employ a Random Forest Regressor, where each leaf node is modeled using a Gaussian distribution based on the mean and variance of the closing prices that fall within each node. The predictions from each tree in the forest are then combined using a weighted optimization. The weights are determined by Maximum Likelihood Estimation (MLE), maximizing the log-likelihood of the observed data. This method assigns higher weights to trees that demonstrate better predictive performance, leading to more accurate forecasts of stock closing prices.")

# Tab 2: Visualizing Technical Indicators
with tab2:
    st.subheader("Technical Indicator Visualizations")

    st.write(
        "RSI (Relative Strength Index): Identifies overbought or oversold market conditions, helping to signal potential reversals.")

    st.write(
        "WPR (Williams %R): Measures overbought and oversold levels based on price extremes, aiding in timing entry and exit points.")

    st.write(
        "Stochastic Oscillator: Compares a particular closing price to a range of prices over time, signaling potential trend reversals.")

    st.write(
        "MACD (Moving Average Convergence Divergence): Indicates momentum by comparing short and long-term price trends, useful for detecting buy/sell signals.")

    st.write(
        "MA50 & MA200 (Moving Averages): Trend-following indicators that smooth price data over 50 and 200 days, respectively, to show the general direction of the market.")

    # Allow the user to select which indicator to visualize
    indicator = st.selectbox('Select Technical Indicator to visualize:',
                             ['RSI', 'MACD', 'Stochastic Oscillator', 'Williams %R', 'MA50', 'MA200'])

    # Plot selected indicator
    fig, ax = plt.subplots()

    if indicator == 'RSI':
        ax.plot(df.index, df['rsi'], label='RSI')
        ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    elif indicator == 'MACD':
        ax.plot(df.index, df['macd'], label='MACD')
        ax.plot(df.index, df['signal_line'], label='Signal Line')
    elif indicator == 'Stochastic Oscillator':
        ax.plot(df.index, df['slowk'], label='Slow %K')
        ax.plot(df.index, df['slowd'], label='Slow %D')
        ax.axhline(80, color='red', linestyle='--', label='Overbought (80)')
        ax.axhline(20, color='green', linestyle='--', label='Oversold (20)')
    elif indicator == 'Williams %R':
        ax.plot(df.index, df['wpr'], label='Williams %R')
        ax.axhline(-20, color='red', linestyle='--', label='Overbought (-20)')
        ax.axhline(-80, color='green', linestyle='--', label='Oversold (-80)')
    # elif indicator == 'MFI':
    # ax.plot(df.index, df['mfi'], label='MFI')
    # ax.axhline(80, color='red', linestyle='--', label='Overbought (80)')
    # ax.axhline(20, color='green', linestyle='--', label='Oversold (20)')
    elif indicator == 'MA50':
        ax.plot(df.index, df['ma50'], label='MA50')
    elif indicator == 'MA200':
        ax.plot(df.index, df['ma200'], label='MA200')

    # Customizing the plot
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{indicator} Value')
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.subheader("Method 1 : Random Forest and GMM")

    # Streamlit selectbox for number of features
    (
        predicted_prices,
        predicted_samples,
        interval_probs,
        predicted_probs,
        interval_80_lower,
        interval_80_upper,
        heat_dates,
        heat_prices,
        prob_matrix,
        display_top_prices,
        display_actuals,
        gmm_avg_pred_prices
    ) = run_gmm_model(
        rf_model_base,
        X_test_selected,
        y_test,
        test_df,
        delta=1.0,
        pred_interval_level=0.80,
        show_plots=True
    )
    st.subheader("Evaluation Metrics")
    st.subheader("Model performance metrics : Relative Probability Error")
    # Model evaluation
    evaluate_model(predicted_prices, y_test)

    st.subheader("Model performance metrics : Average Predicted Price")
    evaluate_model(gmm_avg_pred_prices, y_test)

    # Table summary
    summary_df = pd.DataFrame({
        "Date": test_df['Date'].dt.strftime('%Y-%m-%d'),
        "Predicted Price": predicted_prices,
        "Probability in ±$1": interval_probs,
        "Relative Probability": predicted_probs,
        "80% Lower": interval_80_lower,
        "80% Upper": interval_80_upper
    })
    st.subheader("Prediction Intervals from GMM")
    st.dataframe(
        summary_df.style.format({
            "Predicted Price": "{:.2f}",
            "Probability in ±$1": "{:.2%}",
            "Relative Probability": "{:.2%}",
            "80% Lower": "{:.2f}",
            "80% Upper": "{:.2f}"
        })
    )

    # --- Generate and display sentences in requested style ---
    st.subheader("Daily Prediction Summary")
    for i in range(len(predicted_prices)):
        date = test_df['Date'].dt.strftime('%Y-%m-%d').iloc[i]
        pred = predicted_prices[i]
        prob = interval_probs[i]
        lower = pred - 1.0
        upper = pred + 1.0
        int80_low = interval_80_lower[i]
        int80_high = interval_80_upper[i]
        st.markdown(
            f""" On {date}: </b> We predict the closing price will be <b>${pred:.2f} 

            There is a  {prob:.0%} probability  the actual price will fall within  ${lower:.2f}  and  ${upper:.2f} , and an  80% probability  it will be between  ${int80_low:.2f}  and  ${int80_high:.2f}.""",
            unsafe_allow_html=True
        )

    # Plot the full history + predictions
    plot_results(train_df, test_df, y_test, gmm_avg_pred_prices, predicted_prices)
    plot_results_sub(test_df, y_test, gmm_avg_pred_prices, predicted_prices)

with tab4:
    st.subheader("Method 2: Optimized Random Forest with MLE")

    # Prepare features
    X_train = train_df[all_features].values
    y_train = train_df['Close'].values
    X_test = test_df[all_features].values
    y_test = test_df['Close'].values

    # Train RF and feature selection on train set only


    # Call methodology
    paper_methodology(
        rf_model_base,
        X_train_selected, y_train,
        X_test_selected, y_test,
        train_df, test_df
    )

st.caption("This dashboard is for educational purposes and not investment advice.")

