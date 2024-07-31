from sklearn.datasets import fetch_california_housing
import pandas as pd
import statsmodels.api as sm
import copy

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
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    important_features = list(stepwise_selection(y, X))
    print(sm.OLS(y, sm.add_constant(X[important_features])).fit().summary())