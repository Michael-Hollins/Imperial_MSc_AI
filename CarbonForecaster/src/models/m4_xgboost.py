# XGBoost 
import sys
sys.path.append('/vol/bitbucket/mdh323/Imperial_MSc_AI/CarbonForecaster/src')
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import xgboost as xgb
from stratified_sampling import get_peer_groups_and_folds
import cupy as cp
from sklearn.metrics import mean_absolute_percentage_error

hyper_parameter_tune = False


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
       'activity_sector'], minimum_firms_per_fold=10, k_folds=5, seed_num=42, verbose=True)
target_var = 's1_co2e'
categorical_vars = ['peer_group', 'cc', 'fold']

# Subset the dataframe to variables of interest
s1_data = s1_data[[target_var] + categorical_vars + numeric_x_vars]

# One-hot encode the categoricals
s1_data = pd.get_dummies(s1_data, columns=['peer_group', 'cc'])

ps = PredefinedSplit(test_fold=s1_data['fold'] - 1)  # Subtracting 1 to make it zero-indexed
for i, (train_index, test_index) in enumerate(ps.split()):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

# Split the data
X = s1_data.drop(columns=[target_var, 'fold']).values
y = s1_data[target_var].values

if hyper_parameter_tune == True:
       
       # Define the model
       xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, tree_method = "hist")

       # Define the parameter grid
       param_grid = {
           'learning_rate': [0.01, 0.1, 0.3],
           'max_depth': [4, 6, 8],
           'eta': [0.1, 0.3, 0.5],
           'subsample': [0.8, 0.9, 1.0],
           'colsample_bytree': [0.8, 0.9, 1.0],
           'gamma': [0, 0.1, 0.2],
           'min_child_weight': [1, 3, 5]
       }

       # Set up GridSearchCV with PredefinedSplit
       grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                                  cv=ps, scoring='neg_mean_absolute_error', 
                                  verbose=1, n_jobs=-1)

       # Fit the model
       grid_search.fit(X, y)

       # Output the best parameters and best score
       best_params = grid_search.best_params_
       print("Best parameters found: ", grid_search.best_params_)
       print("Best MAE found: ", -grid_search.best_score_)
else:
       best_params = {'colsample_bytree': 0.8, 'gamma': 0, 'eta':0.1, 'learning_rate': 0.1, 'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.9}

# Now we have our best parameters, model using the GPU
X_cp = cp.array(X)
y_cp = cp.array(y)

# Create DMatrix with CuPy arrays for GPU training
dtrain = xgb.DMatrix(X_cp, label=y_cp)

# Train the final model with the best parameters on GPU
final_model = xgb.train(params={**best_params, 'objective': 'reg:squarederror',
                                'tree_method': 'hist',
                                'device': 'cuda'},
                        dtrain=dtrain)


# Finally, get the average error by iterating over each fold
folds = s1_data['fold'].unique()
global_errors = list()

# Iterate over each fold
for fold in folds:
       train_data = s1_data[s1_data['fold'] != fold]
       test_data = s1_data[s1_data['fold'] == fold]
       
       X_train = train_data.drop(columns=[target_var, 'fold']).values
       y_train = train_data[target_var].values
       X_test = test_data.drop(columns=[target_var, 'fold']).values
       y_test = test_data[target_var].values
       
       # Move data to the GPU
       X_train_cp = cp.array(X_train)
       y_train_cp = cp.array(y_train)
       X_test_cp = cp.array(X_test)
       
       # Create the DMAtrix for the XGBoost model
       dtrain = xgb.DMatrix(X_train_cp, label=y_train_cp)
       dtest = xgb.DMatrix(X_test_cp)
       
       # Train the model using the best parameters on the training fold
       final_model = xgb.train(params={**best_params, 'objective': 'reg:squarederror',
                               'tree_method': 'hist',
                               'device': 'cuda'},
                                dtrain=dtrain)
       
       # Predict on the test fold
       y_pred = final_model.predict(dtest)
       # Compute the error for this fold
       fold_error = mean_absolute_percentage_error(y_test, y_pred)
       global_errors.append(fold_error)
       print(f"Fold {fold} MAPE: {fold_error:.4f}")
       
# Compute the average error across all folds
average_global_error = np.mean(global_errors)
print(f"Average MAPE across all folds: {average_global_error:.4f}")