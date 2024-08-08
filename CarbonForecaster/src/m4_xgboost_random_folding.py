import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, PredefinedSplit, GroupKFold
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from stratified_sampling import get_peer_groups_and_folds
import pickle

SEED = 42
RUN_TRAINING = False

MyGroupKFold = GroupKFold(n_splits=10)

#############################
# Load and pre-process data
#############################

data = pd.read_csv('data/ftse_world_allcap_clean.csv')

# Define our feature variables
fundamental_vars = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', 'receivables', 'net_ppe', 'capex', 'intangible_assets', 'lt_debt']
financial_year_dummies = [x for x in data.columns if "financial_year_" in str(x)]
econ_type_dummies = [x for x in data.columns if 'cc_classification_' in str(x)]
econ_sectors_dummies = [x for x in data.columns if 'econ_sector_' in str(x)]

# Select the dataset
y_choice = ['s1_co2e']
x_choice = fundamental_vars + financial_year_dummies + econ_type_dummies + econ_sectors_dummies
id_choice = ['instrument']
data_choice = data.copy()
df = data_choice.dropna(subset=y_choice + x_choice + id_choice)

# Core data split: 90-10 train-test split
X, y, group = df[x_choice].values, df[y_choice].values, df[id_choice].values
random_test_fold = np.random.randint(1,10)
train_inds, test_inds = next(GroupShuffleSplit(test_size = .1, random_state =SEED).split(X,y,group))
df = df[x_choice + y_choice + id_choice].dropna()
X_train, X_test, y_train, y_test, ID_train, ID_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds], group[train_inds], group[test_inds]

print("Dataset has {} entries and {} features".format(*df.shape)) # Includes X plus the target and id variables and fold number
print("Dataset Train has {} entries and {} features".format(*X_train.shape))
print("Dataset Test has {} entries and {} features".format(*X_test .shape))

# Format the data for XGB
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=x_choice)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=x_choice)

#############################
# Train the XGB Model
#############################
if RUN_TRAINING:
    params_xgb = { 
        # Parameters to tune.
        'max_depth':6,
        'min_child_weight': 1,
        'eta':.3,
        'lambda': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        # Other parameters
        'objective':'reg:squarederror',
        'eval_metric': 'mae'}

    gridsearch_parameters = {
        'max_depth': list(range(6, 12)),
        'min_child_weight': list(range(1, 9)),
        'eta': [0.00001,0.0001,0.005,0.001,0.05,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'lambda': [0.00001, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
        'subsample': [i/10. for i in range(5,11)],
        'colsample_bytree': [i/10. for i in range(5,11)],
    }

    best_params = {}  
    best_boost_rounds = 0  

    for parameter in gridsearch_parameters:
        
        min_eval_score = float('inf')
        
        for val_to_try in gridsearch_parameters[parameter]:
            # Print the task
            print('CV for parameter {} with value {}'.format(parameter, val_to_try))
            
            # Update the parameter value to cross-validate
            params_xgb[parameter] = val_to_try
            
            cv_results = xgb.cv(params=params_xgb, dtrain=dtrain, num_boost_round=1000, nfold=10, folds=list(MyGroupKFold.split(X_train,y_train,ID_train)), early_stopping_rounds=50, seed=SEED)
            # Update best MAE
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds".format(np.round(mean_mae), boost_rounds))
            if mean_mae < min_eval_score:
                min_eval_score = mean_mae
                best_params[parameter] = val_to_try
                best_boost_rounds = boost_rounds
            
        params_xgb.update(best_params)

    with open('params_xgb.pickle', 'wb') as handle:
        pickle.dump((params_xgb, best_boost_rounds), handle)

#############################
# Fit the XGB Model
#############################
params_xgb, optimal_boost_rounds = pickle.load(open('params_xgb.pickle', 'rb'))

final_model = xgb.train(
    params=params_xgb, 
    dtrain=dtrain, 
    num_boost_round=optimal_boost_rounds
)

# Save the final model
final_model.save_model('final_xgb_model.ubj')

# Make predictions on the test set
y_pred = final_model.predict(dtest)

# Evaluate performance on the test set
test_mae = mean_absolute_error(y_test, y_pred)
test_mape = mean_absolute_percentage_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f'Test MAE: {test_mae}')
print(f'Test MAPE: {test_mape}')
print(f'Test R^2: {test_r2}')