# src/gmm_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# seaborn is optional (rugplot only)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def gmm_cdf(gmm: GaussianMixture, x: np.ndarray) -> np.ndarray:
    """
    CDF of a fitted GaussianMixture evaluated at x (vectorized).
    """
    x = np.asarray(x, dtype=float)
    cdf_vals = np.zeros_like(x, dtype=float)
    w = gmm.weights_.flatten()
    m = gmm.means_.flatten()
    v = gmm.covariances_.flatten()
    for wi, mi, vi in zip(w, m, v):
        cdf_vals += wi * norm.cdf(x, loc=mi, scale=np.sqrt(vi))
    return cdf_vals


def run_gmm_model(
    best_rf_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    max_components: int = 3,       # tighter & faster than 14
    grid_size: int = 600,          # resolution for PDF/CDF grid
    delta: float = 1.0,            # ±$ window around mode
    pred_interval_level: float = 0.80,
    show_plots: bool = False
):
    """
    Fit a small GMM (by BIC) to per-tree predictions for each test day.
    Returns:
      - predicted_prices (mode of GMM per day)
      - predicted_samples (list of arrays of per-tree predictions per day)
      - interval_probs (P(|y - mode| <= delta))
      - rel_mode_probs (normalized prob at chosen mode among component means)
      - interval_80_lower, interval_80_upper (central prediction interval bounds)
      - heat_dates, heat_prices, prob_matrix (for heatmap over last 7 days)
      - display_top_prices, display_actuals (last 7 days)
      - gmm_avg_pred_prices (expected value of GMM per day)
      - heatmap_predicted_probs (probability at displayed top price, last 7 only)
    """
    num_days = int(X_test.shape[0])

    top_prices = []                # mode per day (component mean with highest normalized prob)
    predicted_samples = []         # list of arrays: RF tree predictions for each day
    interval_probs = []            # prob within ±delta around mode
    interval_low = []              # lower bound of central PI
    interval_high = []             # upper bound of central PI
    best_Ks = []                   # selected number of components per day
    avg_pred_prices = []           # expected value (weighted mean) of GMM per day
    gmm_variances = []             # variance of the mixture (for optional display)
    rel_mode_probs = []            # normalized probability at the chosen mode

    for i in range(num_days):
        # ----------------------------
        # 1) Collect tree predictions
        # ----------------------------
        samples = np.array([est.predict([X_test[i]])[0] for est in best_rf_model.estimators_], dtype=float).reshape(-1, 1)
        predicted_samples.append(samples.flatten())

        # ----------------------------
        # 2) Fit best-BIC GMM
        # ----------------------------
        best_gmm, best_bic, best_K = None, np.inf, None
        bic_scores = []
        for K in range(1, max_components + 1):
            g = GaussianMixture(n_components=K, random_state=42, n_init=1).fit(samples)
            bic = g.bic(samples)
            bic_scores.append(bic)
            if bic < best_bic:
                best_bic, best_gmm, best_K = bic, g, K
        best_Ks.append(best_K)

        # BIC curve (optional)
        if show_plots:
            date_str = pd.to_datetime(test_df.iloc[i]["Date"]).strftime("%Y-%m-%d")
            fig, ax = plt.subplots()
            ax.plot(range(1, max_components + 1), bic_scores, marker='o', linestyle='-')
            ax.plot(best_K, bic_scores[best_K - 1], 'ro', label=f"Selected: {best_K}")
            ax.set_title(f'BIC Curve – {date_str}')
            ax.set_xlabel('Number of Components (K)')
            ax.set_ylabel('BIC')
            ax.legend()
            st.pyplot(fig)

        # -------------------------------------------------------
        # 3) Mixture mean/variance and mode among component means
        # -------------------------------------------------------
        w = best_gmm.weights_.flatten()
        m = best_gmm.means_.flatten()
        v = best_gmm.covariances_.flatten()

        gmm_mean = float(np.sum(w * m))
        gmm_var = float(np.sum(w * (v + (m - gmm_mean) ** 2)))
        avg_pred_prices.append(gmm_mean)
        gmm_variances.append(gmm_var)

        x_grid = np.linspace(samples.min(), samples.max(), int(grid_size)).reshape(-1, 1)
        x_points = x_grid.flatten()

        # Normalize probabilities evaluated at component means
        means = np.round(m, 2)
        dens = np.exp(best_gmm.score_samples(means.reshape(-1, 1)))
        probs = dens / dens.sum()
        idx = np.argsort(probs)[::-1]
        best_mean = float(means[idx][0])   # mode proxy: component mean with highest normalized prob
        best_prob = float(probs[idx][0])
        top_prices.append(best_mean)
        rel_mode_probs.append(best_prob)

        # Probability within ±delta around the mode
        interval_prob = 0.0
        for wi, mi, vi in zip(w, m, v):
            sigma = np.sqrt(vi)
            interval_prob += wi * (norm.cdf(best_mean + delta, mi, sigma) -
                                   norm.cdf(best_mean - delta, mi, sigma))
        interval_probs.append(float(interval_prob))

        # Central prediction interval (e.g., 80%)
        cdf_vals = gmm_cdf(best_gmm, x_points)
        low_p = (1.0 - pred_interval_level) / 2.0
        high_p = 1.0 - low_p
        low_idx = int(np.searchsorted(cdf_vals, low_p))
        high_idx = int(np.searchsorted(cdf_vals, high_p))
        lower_bound = float(x_points[low_idx]) if low_idx < len(x_points) else float(x_points[0])
        upper_bound = float(x_points[high_idx]) if high_idx < len(x_points) else float(x_points[-1])
        interval_low.append(lower_bound)
        interval_high.append(upper_bound)

        # Distribution plot (optional)
        if show_plots:
            date_str = pd.to_datetime(test_df.iloc[i]["Date"]).strftime("%Y-%m-%d")
            log_pdf = best_gmm.score_samples(x_grid)
            pdf = np.exp(log_pdf)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(samples.flatten(), bins=20, density=True,
                    alpha=0.4, color='skyblue', edgecolor='k', label='Tree predictions (hist)')
            ax.plot(x_points, pdf, linewidth=2, label='GMM Fit (PDF)')

            if _HAS_SNS:
                try:
                    sns.rugplot(samples.flatten(), ax=ax, height=0.06, lw=1, alpha=0.5, zorder=2, clip_on=False)
                except Exception:
                    pass

            ax.axvline(best_mean, color='green', linestyle='--', linewidth=2, label=f'Mode {best_mean:.2f}')
            ax.axvline(gmm_mean, color='purple', linestyle=':', linewidth=2, label=f'Avg {gmm_mean:.2f}')
            ax.axvline(float(y_test[i]), color='blue', linestyle='--', linewidth=2, label=f'Actual {float(y_test[i]):.2f}')
            ax.axvspan(lower_bound, upper_bound, color='orange', alpha=0.15,
                       label=f'{int(pred_interval_level * 100)}% interval')

            ax.set_title(f'GMM Distribution – {date_str} (K={best_K})')
            ax.set_xlabel('Price'); ax.set_ylabel('Density')
            ax.legend()
            st.pyplot(fig)

    # --------------------------------------------
    # Heatmap data for last 7 days (display only)
    # --------------------------------------------
    num_display = min(7, num_days)
    display_dates = test_df['Date'].dt.strftime('%Y-%m-%d').values[-num_display:]
    display_top_prices = np.array(top_prices, dtype=float)[-num_display:]
    display_actuals = np.array(y_test, dtype=float)[-num_display:]

    all_probs_for_heatmap = []
    heatmap_predicted_probs = []
    for i in range(num_days - num_display, num_days):
        samples2d = predicted_samples[i].reshape(-1, 1).astype(float)
        K = best_Ks[i]
        g = GaussianMixture(n_components=K, random_state=42, n_init=1).fit(samples2d)
        means = np.round(g.means_.flatten(), 2)
        dens = np.exp(g.score_samples(means.reshape(-1, 1)))
        probs = dens / dens.sum()
        for m_i, p_i in zip(means, probs):
            all_probs_for_heatmap.append({
                'Date': display_dates[i - (num_days - num_display)],
                'Price': float(m_i),
                'Probability': float(p_i)
            })
        # probability at displayed top price (closest component mean)
        nearest_idx = int(np.argmin(np.abs(means - display_top_prices[i - (num_days - num_display)])))
        heatmap_predicted_probs.append(float(probs[nearest_idx]))

    df_pp = pd.DataFrame(all_probs_for_heatmap)
    pivot = df_pp.pivot(index='Price', columns='Date', values='Probability').fillna(0.0)
    heat_dates = pivot.columns.tolist()
    heat_prices = pivot.index.tolist()
    prob_matrix = pivot.values

    # ----------------
    # Final returns
    # ----------------
    return (
        np.asarray(top_prices, dtype=float),       # predicted_prices (mode)
        predicted_samples,                          # list of 1D arrays (per-day tree preds)
        np.asarray(interval_probs, dtype=float),
        np.asarray(rel_mode_probs, dtype=float),    # probability at chosen mode
        np.asarray(interval_low, dtype=float),
        np.asarray(interval_high, dtype=float),
        heat_dates,                                  # heatmap x
        heat_prices,                                 # heatmap y
        prob_matrix,                                 # heatmap z
        display_top_prices,                          # last-7 modes (for red X)
        display_actuals,                             # last-7 actuals (green dots)
        np.asarray(avg_pred_prices, dtype=float),    # GMM expected value per day
        np.asarray(heatmap_predicted_probs, dtype=float)  # last-7 only (for hover)
    )
