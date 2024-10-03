import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy
from stratified_sampling import get_peer_groups_and_folds

# Stepwise regression    
def forward_step(y, X, included_features, inclusion_threshold, verbose):
    """
    Perform a forward step in stepwise regression by adding the most significant feature.

    Parameters:
    y (array-like): The target variable.
    X (DataFrame): The input features.
    included_features (set): The currently included features.
    inclusion_threshold (float): The p-value threshold for including a feature.
    verbose (bool): If True, print details of the forward step.

    Returns:
    set: The updated set of included features with the most significant new feature added.
    """
    candidate_features = set(X.columns) - included_features
    p_values = pd.Series(index=candidate_features, dtype=float)
    
    for candidate_feature in candidate_features:
        expanded_features = list(included_features) + [candidate_feature]
        model = sm.OLS(y, sm.add_constant(X[expanded_features])).fit()
        p_values[candidate_feature] = model.pvalues.loc[candidate_feature]
    
    if p_values.min() < inclusion_threshold:
        best_feature = p_values.idxmin()
        included_features.add(best_feature)
        if verbose:
            print(f'Add {best_feature} with p-val {p_values.min():.6}')
            
    return included_features

def backward_step(y, X, included_features, exclusion_threshold, verbose):
    """
    Perform a backward step in stepwise regression by removing the least significant feature.

    Parameters:
    y (array-like): The target variable.
    X (DataFrame): The input features.
    included_features (set): The currently included features.
    exclusion_threshold (float): The p-value threshold for excluding a feature.
    verbose (bool): If True, print details of the backward step.

    Returns:
    set: The updated set of included features with the least significant feature removed.
    """
    model = sm.OLS(y, sm.add_constant(X[list(included_features)])).fit()
    p_values = model.pvalues.iloc[1:]
    
    if p_values.max() > exclusion_threshold:
        worst_feature = p_values.idxmax()
        included_features.remove(worst_feature)
        if verbose:
            print(f'Remove {worst_feature} with p-val {p_values.max():.6}')
            
    return included_features

def stepwise_selection(y, X, included_features=set(), inclusion_threshold=0.01, exclusion_threshold=0.05, verbose=True):
    """
    Perform stepwise regression using both forward and backward steps.

    Parameters:
    y (array-like): The target variable.
    X (DataFrame): The input features.
    included_features (set): The initially included features.
    inclusion_threshold (float): The p-value threshold for including a feature.
    exclusion_threshold (float): The p-value threshold for excluding a feature.
    verbose (bool): If True, print details of each step.

    Returns:
    set: The final set of selected features after stepwise regression.
    """
    while True:
        initial_features = copy.deepcopy(included_features)
        forward_reg_features = forward_step(y=y, X=X, included_features=included_features, inclusion_threshold=inclusion_threshold, verbose=verbose)
        backward_reg_features = backward_step(y=y, X=X, included_features=forward_reg_features, exclusion_threshold=exclusion_threshold, verbose=verbose)
        
        if initial_features == backward_reg_features:
            final_features = initial_features
            if verbose:
                print("Final selected features:", final_features)
                print("Ignored features:", set(X.columns) - final_features)
            break
        included_features = backward_reg_features

    return final_features

if __name__ == '__main__':  
    data_file_path = 'data/ftse_world_allcap_clean.csv'
    SEED = 42; np.random.seed(SEED)
    
    data = pd.read_csv(data_file_path)

    # Let's split the dataset into custom peer groups
    sectors = ['econ_sector', 'business_sector', 'industry_group_sector', 'industry_sector', 'activity_sector']
    data = get_peer_groups_and_folds(data, 's1_and_s2_co2e', sector_features=sectors, \
        minimum_firms_per_fold=5, k_folds=10, seed_num=SEED, verbose=True)
    data = pd.concat([data, pd.get_dummies(data['peer_group'], prefix='peer_group', drop_first=True)], axis=1)

    # Feature variables
    fundamentals = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', \
                    'receivables', 'net_ppe', 'capex', 'intangible_assets', 'lt_debt']
    emission_scores = ['policy_emissions_score', 'target_emissions_score']
    financial_year_dummies = [x for x in data.columns if "financial_year_" in str(x)]
    econ_type_dummies = [x for x in data.columns if 'cc_classification_' in str(x)]
    peer_dummies = [x for x in data.columns if 'peer_group_' in str(x)]
    
    # Define the core dataframe 
    x_choice = fundamentals + emission_scores + financial_year_dummies + econ_type_dummies + peer_dummies
    y_choice = ['s1_and_s2_co2e']
    id_choice = ['instrument']
    data_choice = data.copy()
    df = data_choice[id_choice + y_choice + ['fold'] + x_choice].dropna().reset_index(drop=True)
    
    # Pick a random fold for our test fold
    random_test_fold = np.random.randint(1, 10)
    print(f'\nThe random test fold is {random_test_fold}')

    train_inds = np.array(df.index[df['fold'] != random_test_fold])
    test_inds = np.array(df.index[df['fold'] == random_test_fold])

    # For the training data, get the indices for 9-fold cross-validation
    custom_folds = []
    training_data = df.iloc[train_inds]
    training_data.reset_index(inplace=True)
    for i in list(training_data['fold'].unique()):
        train_indices = training_data.index[training_data['fold'] != i].tolist()
        test_indices = training_data.index[training_data['fold'] == i].tolist() 
        custom_folds.append((train_indices, test_indices))
        
    X, y, group = df[x_choice], df[y_choice].values, df[id_choice].values
    
    important_features = list(stepwise_selection(y, X))
    print(sm.OLS(y, sm.add_constant(X[important_features])).fit().summary())