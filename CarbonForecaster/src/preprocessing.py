import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from stratified_sampling import get_peer_groups_and_folds

def preprocess_data(file_path='data/ftse_world_allcap_clean.csv', seed=42):
    #############################
    # Load and pre-process data
    #############################
    SEED=seed
    np.random.seed(SEED)
    
    data = pd.read_csv(file_path)
    
    # Define our feature variables
    fundamental_vars = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', 'receivables', 'net_ppe', 'capex', 'intangible_assets', 'lt_debt']
    financial_year_dummies = [x for x in data.columns if "financial_year_" in str(x)]
    econ_type_dummies = [x for x in data.columns if 'cc_classification_' in str(x)]

    # Select the dataset
    y_choice = ['s1_co2e']
    x_choice_pre_peers = fundamental_vars + financial_year_dummies + econ_type_dummies
    id_choice = ['instrument']
    data_choice = data.copy()
    df = data_choice.dropna(subset=y_choice + x_choice_pre_peers + id_choice).reset_index(drop=True)

    # Now splits - try to get some balance with respect to industries/sectors using the custom function
    df = get_peer_groups_and_folds(df, y_choice[0], sector_features=['econ_sector',
        'business_sector', 'industry_group_sector', 'industry_sector',
        'activity_sector'], minimum_firms_per_fold=5, k_folds=10, seed_num=seed, verbose=False)
    df = pd.concat([df, pd.get_dummies(df['peer_group'], prefix='peer_group', drop_first=True)], axis=1)
    peer_dummies = [x for x in df.columns if 'peer_group_' in str(x)]
    x_choice = x_choice_pre_peers + peer_dummies

    # Core data split: 90-10 train-test split
    df = df[x_choice + y_choice + id_choice + ['fold']].dropna().reset_index(drop=True)
    X, y, group = df[x_choice].values, df[y_choice].values, df[id_choice].values
    random_test_fold = np.random.randint(1, 10)
    train_inds = np.array(df.index[df['fold'] != random_test_fold])
    test_inds = np.array(df.index[df['fold'] == random_test_fold])
    X_train, X_test, y_train, y_test, ID_train, ID_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds], group[train_inds], group[test_inds]
    print("Dataset train has {} entries and {} features".format(*X_train.shape))
    print("Dataset test has {} entries and {} features".format(*X_test.shape))

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=x_choice)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=x_choice)

    # Let's scale the data as well for when we need to use scaled variables
    data_train = pd.concat([pd.DataFrame(ID_train, columns=id_choice),
                            pd.DataFrame(X_train, columns=x_choice),
                            pd.DataFrame(y_train, columns=y_choice)], axis=1)

    data_test = pd.concat([pd.DataFrame(ID_test, columns=id_choice),
                           pd.DataFrame(X_test, columns=x_choice),
                           pd.DataFrame(y_test, columns=y_choice)], axis=1)

    # Scale X and Y using Robust scaler due to non-normal distributions
    vars_to_scale = fundamental_vars + y_choice
    data_train_scaled = data_train.copy()
    quantile_range = (25, 75) # this is the default but let's be explicit
    robust_scaler = RobustScaler(quantile_range=quantile_range).fit(data_train_scaled[vars_to_scale]) # compute the scaling to the TRAIN data
    data_train_scaled[vars_to_scale] = robust_scaler.transform(data_train_scaled[vars_to_scale])
    data_test_scaled = data_test.copy()
    data_test_scaled[vars_to_scale] = robust_scaler.transform(data_test_scaled[vars_to_scale]) # scale to train fit

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, ID_train, ID_test = \
        data_train_scaled[x_choice].values, data_test_scaled[x_choice].values, \
        data_train_scaled[y_choice].values, data_test_scaled[y_choice].values, \
        data_train_scaled[id_choice].values, data_test_scaled[id_choice].values

    print("Dataset train scaled has {} entries and {} features".format(*X_train_scaled.shape))
    print("Dataset test scaled has {} entries and {} features".format(*X_test_scaled.shape))
    dtrain_scaled = xgb.DMatrix(X_train_scaled, label=y_train_scaled, feature_names=x_choice)
    dtest_scaled = xgb.DMatrix(X_test_scaled, label=y_test_scaled, feature_names=x_choice)

    # Return all relevant objects
    return fundamental_vars, x_choice, y_choice, id_choice, df, dtrain, dtest, dtrain_scaled, dtest_scaled, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, ID_train, ID_test, random_test_fold, SEED, robust_scaler, train_inds, test_inds

# Usage example:
if __name__ == "__main__":
    fundamental_vars, x_choice, y_choice, id_choice, df, dtrain, dtest, dtrain_scaled, dtest_scaled, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, ID_train, ID_test, random_test_fold, SEED, robust_scaler, train_inds, test_inds = preprocess_data()