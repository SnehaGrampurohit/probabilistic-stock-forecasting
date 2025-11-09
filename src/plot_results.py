import matplotlib.pyplot as plt
import streamlit as st


def plot_results(train_df, test_df, y_test, gmm_avg_pred_prices,predicted_prices):
    """
    Plots the training closing prices, actual test prices,
    and predicted (most-probable) prices.

    Arguments:
      train_df         -- DataFrame with training data (must include 'Date' & 'Close')
      test_df          -- DataFrame with test data (must include 'Date')
      y_test           -- Actual closing prices for the test set
      predicted_prices -- Predicted prices (highest-probability μ_k from GMM)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) Training period
    ax.plot(
        train_df['Date'],
        train_df['Close'],
        label='Training Close Prices',
        color='gray',
        alpha=0.6
    )

    # 2) Actual test prices
    ax.plot(
        test_df['Date'],
        y_test,
        label='Actual Test Prices',
        color='blue',
        marker='o'
    )

    # 3) Predicted (most-probable) prices
    ax.plot(
        test_df['Date'],
        gmm_avg_pred_prices,
        label='Predicted Prices (Estimated Mean of GMM)',
        color='green',
        marker='x'
    )

    ax.plot(
        test_df['Date'],
        predicted_prices,
        label='Predicted Prices (GMM Mode)',
        color='red',
        marker='x'
    )



    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Training vs. Actual vs. Predicted Prices')
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)


def plot_results_sub(test_df, y_test, gmm_avg_pred_prices, predicted_prices):
    """
    Plots the actual test prices and the predicted (most-probable) prices
    over the last 7 days.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Actual prices
    ax.plot(
        test_df['Date'],
        y_test,
        label='Actual Prices (Last 7 Days)',
        color='blue',
        marker='o'
    )

    # Predicted (most-probable) prices
    ax.plot(
        test_df['Date'],
        gmm_avg_pred_prices,
        label='Predicted Prices (Estimated Mean: Last 7 Days)',
        color='green',
        marker='x'
    )

    ax.plot(
        test_df['Date'],
        predicted_prices,
        label='Predicted prices (GMM Mode: Last 7 Days)',
        color='red',
        marker='x'
    )



    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs. Predicted Prices (Last 7 Days)')
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)
