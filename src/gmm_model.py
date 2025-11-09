import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd

def gmm_cdf(gmm, x):
    """Compute the CDF of a Gaussian Mixture Model at value(s) x."""
    cdf_vals = np.zeros_like(x, dtype=float)
    for w, m, s2 in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
        cdf_vals += w * norm.cdf(x, loc=m, scale=np.sqrt(s2))
    return cdf_vals

def run_gmm_model(
    best_rf_model,
    X_test,
    y_test,
    test_df,
    max_components=14,
    grid_size=1000,
    delta=1.0,  # Interval width for ±$ calculation
    pred_interval_level=0.80,  # 80% prediction interval
    show_plots = False
):
    num_days = X_test.shape[0]
    top_prices = []
    predicted_samples = []
    interval_probs = []
    interval_low = []
    interval_high = []
    best_Ks = []  # Store best number of components for each day
    avg_pred_prices = []  # <--- Initialize at top
    gmm_variances = []

    for i in range(num_days):
        # 1) gather tree predictions
        samples = np.array([
            est.predict([X_test[i]])[0]
            for est in best_rf_model.estimators_
        ]).reshape(-1, 1)
        predicted_samples.append(samples.flatten())



        # 2) fit best-BIC GMM
        bic_scores = []

        best_gmm, best_bic = None, np.inf
        best_K = None
        for K in range(1, max_components + 1):
            g = GaussianMixture(n_components=K, random_state=42).fit(samples)
            bic = g.bic(samples)
            bic_scores.append(bic)
            if bic < best_bic:
                best_bic, best_gmm = bic, g
                best_K = K
        best_Ks.append(best_K)

        gmm_mean = np.sum(best_gmm.weights_.flatten() * best_gmm.means_.flatten())
        avg_pred_prices.append(gmm_mean)
        np_avg_pred_prices = np.array(avg_pred_prices)
        gmm_var = np.sum(best_gmm.weights_.flatten() * (
                best_gmm.covariances_.flatten() + (best_gmm.means_.flatten() - gmm_mean) ** 2
        ))
        gmm_variances.append(gmm_var)


        # 3a) Plot BIC curve and mark best K
        date_str = test_df.iloc[i]['Date'].strftime('%Y-%m-%d')
        fig, ax = plt.subplots()
        ax.plot(
            range(1, max_components + 1),
            bic_scores,
            marker='o', linestyle='-'
        )
        # Mark the selected component count
        ax.plot(
            best_K, bic_scores[best_K-1],
            marker='o', color='red', markersize=10,
            label=f"Selected: {best_K}"
        )
        ax.set_title(f'BIC Curve – {date_str}\nSelected Components: {best_K}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('BIC Score')
        ax.legend()
        if show_plots:
            st.pyplot(fig)


        # 3b) Plot distribution (hist + PDF + rugplot)
        x_grid = np.linspace(samples.min(), samples.max(), grid_size).reshape(-1,1)
        x_points = x_grid.flatten()
        log_pdf = best_gmm.score_samples(x_grid)
        pdf = np.exp(log_pdf)

        fig, ax = plt.subplots(figsize=(8, 4))
        # Plot normalized histogram
        ax.hist(
            samples.flatten(), bins=20, density=True,
            alpha=0.4, color='skyblue', edgecolor='k', label='Tree Predictions (hist)'
        )
        # Plot GMM PDF
        ax.plot(
            x_grid, pdf, color='red', linewidth=2, label='GMM Fit (PDF)'
        )
        # Rugplot for tree predictions
        sns.rugplot(samples.flatten(), ax=ax, color='gray', height=0.06, lw=1, alpha=0.5, zorder=2, clip_on=False)

        # 4) compute normalized probabilities at each component mean
        means = best_gmm.means_.flatten().round(2)
        weights = best_gmm.weights_
        dens = np.exp(best_gmm.score_samples(means.reshape(-1,1)))
        probs = dens / dens.sum()

        # sort by descending probability
        idx = np.argsort(probs)[::-1]
        sorted_means = means[idx]
        sorted_probs = probs[idx]

        # pick the top (most probable) predicted price
        best_mean = sorted_means[0]
        best_prob = sorted_probs[0]
        top_prices.append(best_mean)

        # PROBABILITY IN INTERVAL [best_mean - delta, best_mean + delta]
        interval_prob = 0.0
        for weight, mu, sigma2 in zip(best_gmm.weights_, best_gmm.means_.flatten(), best_gmm.covariances_.flatten()):
            sigma = np.sqrt(sigma2)
            prob = norm.cdf(best_mean + delta, mu, sigma) - norm.cdf(best_mean - delta, mu, sigma)
            interval_prob += weight * prob
        interval_probs.append(interval_prob)

        # CENTRAL PREDICTION INTERVAL (e.g., 80%) from GMM CDF
        cdf_vals = gmm_cdf(best_gmm, x_points)
        low_p = (1 - pred_interval_level) / 2
        high_p = 1 - low_p

        low_idx = np.searchsorted(cdf_vals, low_p)
        high_idx = np.searchsorted(cdf_vals, high_p)
        lower_bound = x_points[low_idx] if low_idx < len(x_points) else x_points[0]
        upper_bound = x_points[high_idx] if high_idx < len(x_points) else x_points[-1]
        interval_low.append(lower_bound)
        interval_high.append(upper_bound)

        # Overlay predicted & actual on the distribution plot
        ax.axvline(
            best_mean, color='green', linestyle='--', linewidth=2,
            label=f' Most Probable Price (mode = {best_mean:.2f})'
        )

        actual = y_test[i]
        ax.axvline(
            actual, color='blue', linestyle='--', linewidth=2,
            label=f'Actual ({actual:.2f})'
        )


        ax.axvline(
            np_avg_pred_prices[-1], color='purple', linestyle='--', linewidth=2,
            label=f'Average Predicted price ({np_avg_pred_prices[-1]:.2f})'
        )

        ax.axvspan(lower_bound, upper_bound, color='orange', alpha=0.15,
                   label=f'{int(pred_interval_level*100)}% Interval')

        # Overlay variance as annotation on the plot
        ax.text(
            0.98, 0.92,  # X, Y coordinates (relative to axes, top-right)
            f"Variance: {gmm_var:.2f}",
            fontsize=11, color="purple", ha='right', va='center',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='purple')
        )

        ax.set_title(
            f'GMM Distribution – {date_str} (K={best_K})\n'
            f'Relative Prob in [±{delta}]: {interval_prob:.2%}, '
            f'{int(pred_interval_level*100)}% Interval: [{lower_bound:.2f}, {upper_bound:.2f}]'
        )
        ax.set_xlabel('Price')
        ax.set_ylabel('Density')
        ax.legend()
        if show_plots:
            best_mean_fl = round(float(best_mean),2)
            best_prob_fl = round(float(best_prob *100),2)
            st.pyplot(fig)
            st.markdown(
                f"On {date_str}: The most probable price (The mode of the GMM distribution, i.e., the component mean with the highest normalized probability)  is ${best_mean:.2f}. "
            )
            st.markdown(
                f"There is {best_prob_fl}% probability that the actual price will fall within ±$1 of this value. "
            )
            st.markdown(
                f"The Estimated average (the weighted sum of all component means, using their probabilities as weights) is {np_avg_pred_prices[-1]:.2f}. "
            )

            st.markdown(
                f"Distribution Variance (Uncertainty) {gmm_var:.2f}. This means that most predicted prices are typically within about ${np.sqrt(gmm_var):.2f} of the estimated average predicted price. "
            )




    # 1. Prepare data for last 7 days (or all days, adjust as needed)
    num_display = min(7, num_days)
    display_dates = test_df['Date'].dt.strftime('%Y-%m-%d').values[-num_display:]
    display_top_prices = np.array(top_prices)[-num_display:]
    display_actuals = y_test[-num_display:]

    # Reconstruct component probabilities for the heatmap
    all_probs_for_heatmap = []
    for i in range(num_days - num_display, num_days):
        samples = predicted_samples[i]
        samples2d = samples.reshape(-1, 1)
        best_K = best_Ks[i]
        best_gmm = GaussianMixture(n_components=best_K, random_state=42).fit(samples2d)
        means = best_gmm.means_.flatten().round(2)
        dens = np.exp(best_gmm.score_samples(means.reshape(-1, 1)))
        probs = dens / dens.sum()
        for mean, prob in zip(means, probs):
            all_probs_for_heatmap.append({
                'Date': display_dates[i - (num_days - num_display)],
                'Price': mean,
                'Probability': prob
            })

    df_pp = pd.DataFrame(all_probs_for_heatmap)
    pivot = df_pp.pivot(index='Price', columns='Date', values='Probability').fillna(0)
    heat_dates = pivot.columns.tolist()
    heat_prices = pivot.index.tolist()
    prob_matrix = pivot.values

    # Find predicted price probabilities (for red X hover)
    predicted_probs = []
    for i, (date, price) in enumerate(zip(heat_dates, display_top_prices)):
        price_idx = np.argmin(np.abs(np.array(heat_prices) - price))
        predicted_probs.append(prob_matrix[price_idx, i])

    # 2. Build Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=heat_dates,
        y=heat_prices,
        z=prob_matrix,
        colorscale='YlGnBu',
        colorbar=dict(title='Probability'),
        zmin=0,
        zmax=prob_matrix.max()
    ))

    fig.add_trace(go.Scatter(
        x=heat_dates,
        y=display_top_prices,
        mode='markers',
        marker=dict(symbol='x', size=16, color='red'),
        customdata=np.round(predicted_probs, 4),
        hovertemplate=(
            "Date: %{x}<br>"
            "Predicted Price: %{y:.2f}<br>"
            "Probability: %{customdata:.2%}<extra></extra>"
        ),
        name='GMM Relative Probability'
    ))

    fig.add_trace(go.Scatter(
        x=heat_dates,
        y=display_actuals,
        mode='markers',
        marker=dict(symbol='circle', size=10, color='green'),
        name='Actual Values'
    ))

    fig.add_trace(go.Scatter(
        x=heat_dates,
        y=avg_pred_prices,
        mode='lines+markers',
        marker=dict(symbol='triangle-up', size=14, color='purple'),
        line=dict(color='purple', dash='dot'),
        name='GMM Avg Predicted'
    ))

    fig.update_layout(
        title="Probability Distribution Heatmap for Last 7 Days<br>(Red ×: Predicted, Black ○: Actual)",
        xaxis=dict(title="Date", tickangle=-45),
        yaxis=dict(title="Price"),
        margin=dict(l=60, r=20, t=80, b=80),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if show_plots:
        st.plotly_chart(fig, use_container_width=True)

    # Keep only the average means for the days displayed in the heatmap
    gmm_avg_pred_prices = pd.Series(avg_pred_prices[-len(heat_dates):])



    # Return extra info for tab5 heatmap, not just summary stats
    return (
        np.array(top_prices),
        predicted_samples,
        np.array(interval_probs),
        np.array(interval_low),
        np.array(interval_high),
        heat_dates,         # NEW
        heat_prices,        # NEW
        prob_matrix,        # NEW
        display_top_prices, # NEW
        display_actuals,    # NEW
        gmm_avg_pred_prices
    )


"""2. Why is the “Relative Probability” Error Higher?

A. The Mode Can Miss Outliers
	•	The “relative probability” value is the mode—it picks the price where the GMM assigns the highest probability, but ignores all other plausible outcomes.
	•	If data has multiple plausible price clusters (is multi-modal), or if the true price falls between modes, the mode prediction can be off.
	•	The mode is not robust to the shape of the distribution. For skewed or multi-peaked distributions, it may pick a peak that is not actually closest to the true value.

B. The Mean (Average Predicted Price) Balances All Possibilities
	•	The mean (average predicted price) considers all the component means and their probabilities.
	•	Even if the distribution is wide or has multiple peaks, the mean provides a central tendency, often closer to the “center of mass” of the true values.

C. Typical in Probabilistic Models
	•	In general, when distributions are symmetric and unimodal, the mode and mean are close, and errors are similar.
	•	In real-world financial data, distributions are often asymmetric and multi-modal, so the mean tends to provide a more stable, lower-error prediction.
	•	Statistically, the mean minimizes the squared error (RMSE).


⸻

4. Analogy
	•	Imagine rolling a dice where the sides are not equally likely (weighted dice).
	•	Mode: Always bet on the side most likely to appear (could be 6, but sometimes 3 appears just as often).
	•	Mean: Bet on the average value across many rolls; over time, this is more accurate.

⸻

5. Conclusion

The higher error for “Relative Probability” simply reflects the fact that in noisy or multi-modal data, always picking the “most likely” single value is less robust than taking the average of all likely values. The mean is statistically optimal for minimizing average error, which is why you see lower RMSE and MAE there.
"""