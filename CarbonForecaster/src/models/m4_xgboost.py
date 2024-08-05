# XGBoost 

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from clean_raw_data import get_emissions_intensity_cols
from eda import audit_value_types
from stratified_sampling import get_peer_groups_and_folds

df = pd.read_csv('data/ftse_world_allcap_clean_xgboost.csv')

numeric_x_vars = ['mcap', 'revenue', 'ebit', 'ebitda',
       'gross_profit', 'net_cash_flow', 'assets', 'current_assets',
       'current_liabilities', 'inventories', 'receivables', 'net_ppe',
       'cost_of_revenue', 'capex', 'intangible_assets', 'lt_debt',
       'policy_emissions_score', 'target_emissions_score']
#df[numeric_x_vars] = df[numeric_x_vars].fillna(0)

# Scope 1 modelling
s1_data = get_peer_groups_and_folds(df, 's1_co2e', sector_features=['econ_sector',
       'business_sector', 'industry_group_sector', 'industry_sector',
       'activity_sector'], minimum_firms_per_fold=4, k_folds=5, seed_num=42, verbose=False)
target_var = 's1_co2e'
categorical_vars = ['peer_group', 'cc', 'fold']

# Subset the dataframe to variables of interest
s1_data = s1_data[[target_var] + categorical_vars + numeric_x_vars]

# One-hot encode the categoricals
s1_data = pd.get_dummies(s1_data, columns=['peer_group', 'cc'])

ps = PredefinedSplit(test_fold=s1_data['fold'] - 1)  # Subtracting 1 to make it zero-indexed

# Define the model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Set up GridSearchCV with PredefinedSplit
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=ps, scoring='neg_mean_absolute_error', 
                           verbose=1, n_jobs=-1)

# Separate features and target
X = s1_data.drop(columns=[target_var, 'fold'])
y = s1_data[target_var]

# Fit the model
grid_search.fit(X, y)

# Output the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best MAE found: ", -grid_search.best_score_)