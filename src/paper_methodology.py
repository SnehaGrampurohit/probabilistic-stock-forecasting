import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import scipy.stats as stats
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error


def paper_methodology(
        rf_model, X_train, y_train, X_test, y_test, train_df, test_df
):
    """
    Fits ensemble weights (MLE) on the train set only, evaluates on test set.
    Plots prediction with confidence intervals, Gaussian leaf distributions, and ensemble weights.
    """
    # --- Step 1: Fit Gaussian means/vars to leaves on train set ---
    tree_gaussians = []
    for tree in rf_model.estimators_:
        leaf_ids = tree.apply(X_train)
        leaves_mean_var = {}
        for leaf in np.unique(leaf_ids):
            leaf_samples = y_train[leaf_ids == leaf]
            leaf_mean = np.mean(leaf_samples)
            leaf_var = np.var(leaf_samples)
            leaves_mean_var[leaf] = (leaf_mean, leaf_var)
        tree_gaussians.append(leaves_mean_var)

    # --- Step 2: Gather tree predictions on train set for weight optimization ---
    tree_preds_train = np.array([tree.predict(X_train) for tree in rf_model.estimators_])

    # --- Step 3: Compute variance predictions on train set ---
    var_preds_train = np.zeros(len(X_train))
    for i, row in enumerate(X_train):
        tree_vars = []
        for tree_idx, tree in enumerate(rf_model.estimators_):
            leaf_id = tree.apply([row])[0]
            leaf_mean, leaf_var = tree_gaussians[tree_idx][leaf_id]
            tree_vars.append(leaf_var)
        var_preds_train[i] = np.mean(tree_vars)

    # --- Step 4: MLE Ensemble Weight Optimization on train set ---
    def log_likelihood(alpha, tree_preds, y_true, var_preds):
        weighted_preds = np.dot(alpha, tree_preds)
        residuals = y_true - weighted_preds
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var_preds + 1e-8) + (residuals ** 2) / (var_preds + 1e-8))
        return -log_likelihood

    init_alpha = np.ones(len(rf_model.estimators_)) / len(rf_model.estimators_)
    cons = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1})
    bounds = [(0, 1)] * len(rf_model.estimators_)
    result = minimize(
        log_likelihood, init_alpha, args=(tree_preds_train, y_train, var_preds_train),
        constraints=cons, bounds=bounds, method='SLSQP'
    )
    optimal_alpha = result.x

    # --- Step 5: Apply weights to test set predictions ---
    tree_preds_test = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
    var_preds_test = np.zeros(len(X_test))
    for i, row in enumerate(X_test):
        tree_vars = []
        for tree_idx, tree in enumerate(rf_model.estimators_):
            leaf_id = tree.apply([row])[0]
            leaf_mean, leaf_var = tree_gaussians[tree_idx][leaf_id]
            tree_vars.append(leaf_var)
        var_preds_test[i] = np.mean(tree_vars)
    weighted_preds_test = np.dot(optimal_alpha, tree_preds_test)

    # --- Step 6: Plot results with confidence intervals ---
    fig = plot_results_with_confidence_interval(
        test_df, y_test, weighted_preds_test, var_preds_test
    )
    st.pyplot(fig)

    # --- Step 7: Report metrics on test set ---
    mae = mean_absolute_error(y_test, weighted_preds_test)
    rmse = np.sqrt(mean_squared_error(y_test, weighted_preds_test))
    st.write(f"**Mean Absolute Error (Test):** {mae:.5f}")
    st.write(f"**RMSE (Test):** {rmse:.5f}")

    # --- Step 8: Additional plots: Gaussian leaves & ensemble weights ---
    plot_additional_visualizations(rf_model, optimal_alpha, tree_gaussians, X_test)


# -- Helper plotting functions --
def plot_results_with_confidence_interval(
        test_df, y_test, y_pred, var_preds, alpha=0.05,
        ylabel="Price", title="Stock Closing Price Prediction with Confidence Interval"
):
    z = 1.96 if alpha == 0.05 else abs(np.percentile(np.random.normal(size=10000), [100 * (1 - alpha / 2)]))[0]
    std_pred = np.sqrt(var_preds)
    lower_bound = y_pred - z * std_pred
    upper_bound = y_pred + z * std_pred
    x_vals = test_df['Date'] if 'Date' in test_df.columns else test_df.index

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_test, label='Actual Close Price', color='blue', linewidth=2)
    plt.scatter(x_vals, y_test, color='blue', s=60, marker='o', alpha=0.7)
    plt.plot(x_vals, y_pred, label='Predicted Close Price', color='red', linestyle='dashed', linewidth=2)
    plt.scatter(x_vals, y_pred, color='red', s=60, marker='x', alpha=0.7)
    plt.fill_between(x_vals, lower_bound, upper_bound, color='lightgreen', alpha=0.5,
                     label=f'95% Confidence Interval')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_additional_visualizations(rf_model, optimal_alpha, tree_gaussians, X_to_plot):
    tree_idx = 0
    tree = rf_model.estimators_[tree_idx]
    leaf_ids = tree.apply(X_to_plot)
    unique_leaf_ids = np.unique(leaf_ids)
    plt.figure(figsize=(10, 6))
    for leaf in unique_leaf_ids:
        leaf_mean, leaf_var = tree_gaussians[tree_idx][leaf]
        if leaf_var < 1e-8:
            continue
        x_range = np.linspace(leaf_mean - 3 * np.sqrt(leaf_var), leaf_mean + 3 * np.sqrt(leaf_var), 100)
        plt.plot(x_range, stats.norm.pdf(x_range, leaf_mean, np.sqrt(leaf_var)),
                 label=f'Leaf {leaf} (μ={leaf_mean:.2f}, σ²={leaf_var:.2f})')
    plt.title('Gaussian Distributions at Leaf Nodes (Tree 1)')
    plt.xlabel('Predicted Closing Price')
    plt.ylabel('Probability Density')
    plt.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(optimal_alpha)), optimal_alpha, color='skyblue')
    plt.title('Optimal Ensemble Weights per Tree (from MLE)')
    plt.xlabel('Tree Index')
    plt.ylabel('Assigned Weight')
    plt.xticks(np.arange(len(optimal_alpha)))
    plt.ylim(0, max(1, 1.05 * optimal_alpha.max()))
    for i, w in enumerate(optimal_alpha):
        plt.text(i, w + 0.01, f"{w:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(plt.gcf())


