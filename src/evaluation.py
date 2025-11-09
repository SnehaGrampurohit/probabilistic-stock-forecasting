
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
from sklearn.mixture import GaussianMixture



def evaluate_model(predicted_prices,
                   y_test):
    """
    Evaluates:
      • RMSE & MAE between y_test and predicted_prices (highest-prob price),

    """

    # 1) RMSE & MAE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))
    mae  = mean_absolute_error(y_test, predicted_prices)



    # 3) Display in Streamlit

    st.write(f"• RMSE: {rmse:.4f}")
    st.write(f"• MAE:  {mae:.4f}")
    #st.write(f"• Avg. Log-Likelihood: {avg_log_likelihood:.4f}")