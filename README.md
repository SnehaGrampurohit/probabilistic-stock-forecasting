Probabilistic Stock Forecasting (AMZN)

Inference-only demo of probabilistic stock price forecasting using

Random Forest + RFE feature selection + Gaussian Mixture Model (GMM) distribution

The Streamlit app is deployed here:

App: https://probabilistic-stock-forecasting-grampurohit.streamlit.app/

The key idea is:
models are trained offline, stored as artifacts in the repo, and the Streamlit app only performs fast inference and visualisation.


1. Project Overview

This project forecasts the next-day closing price of Amazon stock (AMZN) and exposes the full predictive distribution rather than just a point forecast.

The workflow is:

	1.	Data & features
	•	Historical AMZN prices from Yahoo Finance.
	•	Technical indicators (RSI, WPR, MACD, MA50, MA200, Stochastic Oscillator, etc.).
	•	Calendar features and lags of the close price.
	
	2.	Offline training (train_offline.py)
	•	Build features on the last 60 months of data.
	•	Use RFE to select the top 5 features.
	•	Train a tuned RandomForestRegressor (500 trees).
	•	Save the trained RF, RFE selector and feature list as artifacts in models/.
	
	3.	Online inference (main_app.py)
	•	Load pre-trained artifacts from models/.
	•	Load a frozen price history snapshot from data/AMZN.csv (no live training).
	•	For the last 7 days:
	•	Collect predictions from each RF tree.
	•	Fit a Gaussian Mixture Model (GMM) to these tree predictions.
	•	Extract:
	•	Mode (most probable price).
	•	Expected value (GMM average).
	•	80% prediction interval.
	•	Probability mass inside ±$1 of the mode.

	
	4.	Evaluation & calibration
	•	Point accuracy: RMSE, MAE.
	•	Distribution quality:
	•	80% PI coverage.
	•	Pinball loss (q = 0.5).
	•	Visual diagnostics (distribution plots, probability heatmap, residuals).


2. App Experience

The Streamlit app has two main tabs:

🔮 Forecast

	•	Headline metrics
	•	RMSE (mode prediction)
	•	MAE (mode prediction)
	•	RMSE (GMM average prediction)
	•	MAE (GMM average prediction)
	•	Last 7 days — interval summary
	•	Date
	•	Predicted price (mode)
	•	Probability that the true close lies within ±$1 of the mode
	•	Relative probability of the mode
	•	80% lower and upper bounds
	•	Plots
	•	Full-history training vs. test vs. predicted prices.
	•	Zoomed last-7-days actual vs. predicted plot.
	•	Advanced diagnostics (slow)
	•	Per-day GMM distribution plots
	
(histogram of tree predictions, GMM PDF, mode, average, actual price, 80% interval).
	•	Probability heatmap for the last 7 days
(price vs. date with probability colour scale, overlaying predicted and actual values).

🧪 Calibration
	•	Headline calibration
	•	80% PI Coverage: fraction of actual closes falling inside the 80% prediction interval.
	•	Pinball loss (q = 0.5): median quantile loss (equals 0.5 × MAE).
	•	Residual diagnostics
	•	Residuals (Actual − GMM average prediction) over the last 7 days.

