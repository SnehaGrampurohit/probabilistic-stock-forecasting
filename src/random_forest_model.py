

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import streamlit as st



def run_random_forest(X_train, y_train, num_features):
    """
    Trains a Random Forest model with RFE for feature selection and performs hyperparameter tuning.
    Arguments:
    X_train -- Feature set for training
    y_train -- Target values for training
    num_features -- Number of features to select (user-defined: 5, 7, or 9)

    Returns:
    best_rf_model -- The trained Random Forest model with the best hyperparameters
    rfe -- The fitted RFE object (for feature selection transformation)
    """

    # Define the feature set (same as in the main program)
    all_features = ['Open',
                    'High',
                    'Low',
                    'Volume',
                    'rsi',
                    'slowk',
                    'slowd',
                    'macd',
                    'ma50',
                    'ma200',
                    'ult_oscillator',
                    'wpr',
                    'day',
                    'month',
                    'quarter',
                    'year',
                    'lag1',
                    'lag2',
                    'lag3']

    # Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Recursive Feature Elimination (RFE) for selecting top features based on user input
    rfe = RFE(estimator=rf, n_features_to_select=num_features)  # Use user-selected number of features
    X_selected = rfe.fit_transform(X_train, y_train)  # Fit and transform the features




    # Get the selected features
    selected_features = [all_features[i] for i in range(len(all_features)) if rfe.support_[i]]
    st.write(f'Selected Features: {selected_features}')

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 30, 50],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 4, 16]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_selected, y_train)

    # Best model from GridSearchCV
    best_rf_model = grid_search.best_estimator_

    # Best parameters and test score from GridSearchCV
    st.write(f'Best Random Forest Parameters: {grid_search.best_params_}')
    best_test_score = grid_search.best_score_
    st.write(f'Best test score from GridSearchCV: {best_test_score}')

    return best_rf_model, rfe  # Return the best model and the fitted RFE object
