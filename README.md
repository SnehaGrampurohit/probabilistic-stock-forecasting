Probability Distribution Prediction of Stock Prices

Random Forest + Gaussian Mixture Models (GMM) & Gaussian-Leaf RF with MLE Optimization

🌐 Live Demo

Experience the full interactive dashboard here:

👉 https://probabilistic-stock-forecasting-grampurohit.streamlit.app/


📌 Overview

This project implements probabilistic forecasting of stock closing prices, based on my Master’s thesis in Data Science.
Instead of predicting a single price, the system models the entire distribution of possible future prices, capturing uncertainty, variance, and confidence intervals.

The application uses:

Method 1 — Random Forest + Gaussian Mixture Model (GMM)
	•	Recursive Feature Elimination (RFE) for feature selection
	•	Random Forest base model
	•	GridSearchCV for hyperparameter tuning
	•	Per-tree predictions → used as samples for GMM
	•	BIC-driven component selection
	•	Outputs:
	•	Most probable price (mode)
	•	Expected price (weighted mean)
	•	±$δ interval probability
	•	80% prediction intervals
	•	Daily GMM distribution plots
	•	Probability heatmaps

Method 2 — Gaussian-Leaf Random Forest + MLE Weight Optimization

	•	Each leaf node modeled as a Gaussian (mean + variance)
  
	•	Variances aggregated to produce predictive uncertainty
  
	•	Maximum Likelihood Estimation (MLE) to find optimal tree weights
  
	•	Outputs:
  
	•	Variance-aware predictions
  
	•	Confidence intervals
  
	•	Leaf Gaussian visualizations
  
	•	Ensemble weight bar charts

The entire system is exposed through a fully interactive Streamlit dashboard.

⸻

📊 Technical Indicators Used

The feature set integrates multiple technical indicators to capture market structure:

	•	RSI (Relative Strength Index)
  
	•	MACD (Moving Average Convergence Divergence)
  
	•	Williams %R
  
	•	Stochastic Oscillator (Slow %K, Slow %D)
  
	•	MA50 & MA200
  
	•	Lag features
  
	•	Date-time decomposition (day, month, year, quarter)

Indicators are computed using pandas-ta for seamless deployment.


📈 Key Features

✔ Distribution-Based Forecasting

Not just a number — full probability distribution over future closing prices.

✔ Uncertainty Quantification

Variance, confidence intervals, ±$δ interval probability.

✔ Explainability Through Visualization
	•	BIC curves
  
	•	Tree prediction distributions
  
	•	Gaussian leaf curves
  
	•	Probability heatmaps
  
	•	Confidence interval bands

✔ Interactive Dashboard

User-controlled:

	•	Number of years of data
  
	•	Number of features
  
	•	Indicator selection
  
	•	Method comparison


🛡️ Disclaimer

This project is developed for academic and educational purposes only.
It does not constitute financial advice or stock market guidance.


📜 License

Distributed under the MIT License.
See LICENSE for details.


📬 Contact

If you have feedback, collaboration requests, or opportunities to discuss:

Sneha Grampurohit
Master of Science — Data Science
Germany

GitHub: https://github.com/SnehaGrampurohit

Live App: https://probabilistic-stock-forecasting-grampurohit.streamlit.app/
