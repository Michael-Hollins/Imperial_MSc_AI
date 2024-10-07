###### ML Pipeline ######


#####################################
############ Libraries ##############
#####################################
import numpy as np
import pandas as pd
from clean_raw_data import get_material_s3_category_reporting
from stratified_sampling import get_peer_groups_and_folds
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pickle
import matplotlib.pyplot as plt


def subset_to_date_range(df, year_lower, year_upper):
    """
    Filters the DataFrame to include only the rows within the specified date range
    and adds one-hot encoded columns for the 'year' column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing at least a 'year' column.
        year_lower (int): The lower bound of the year range (inclusive).
        year_upper (int): The upper bound of the year range (inclusive).

    Returns:
        pandas.DataFrame: A DataFrame filtered to the specified date range with 
        additional one-hot encoded columns for the 'year' column prefixed with 'FY_'.

    Raises:
        ValueError: If `year_lower` is not less than `year_upper`.
        ValueError: If `year_lower` or `year_upper` do not exist in `df['year']`.
    """

    # Validate that year_lower is less than year_upper
    if year_lower >= year_upper:
        raise ValueError(f"'year_lower' ({year_lower}) must be less than 'year_upper' ({year_upper}).")

    # Validate that both year_lower and year_upper exist in df['year']
    if year_lower not in df['year'].values:
        raise ValueError(f"'year_lower' ({year_lower}) does not exist in 'df['year']'.")
    if year_upper not in df['year'].values:
        raise ValueError(f"'year_upper' ({year_upper}) does not exist in 'df['year']'.")

    df = df.loc[(df['year'] >= year_lower) & (df['year'] <= year_upper)]
    df = pd.concat([df, pd.get_dummies(df['year'], prefix='FY', drop_first=True)], axis=1)
    return df


def remove_missing_and_non_positive_values(df, y_choice, verbose=False):
    """
    Removes all rows from the DataFrame where any of the specified columns contain missing values 
    or contain values less than or equal to zero.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        y_choice (list): A list of column names corresponding to the target variables.
        verbose (bool, optional): If True, prints the number of observations dropped due to missing 
                                  values or non-positive values. Default is False.
        x_choice_core (list): A list of column names corresponding to the core feature variables.
        x_choice_extended (list): A list of column names corresponding to the extended feature variables.

    Returns:
        pandas.DataFrame: A DataFrame with rows containing missing or non-positive values in the 
        specified columns removed.

    Notes:
        - The function first removes rows with missing values in the columns specified by `y_choice`, 
          's1_and_s2_co2e', 's1_and_s2_co2e_intensity', and `x_choice_extended`.
        - It then removes rows with non-positive values in the columns specified by `y_choice`, 
          's1_and_s2_co2e', 's1_and_s2_co2e_intensity', and `x_choice_core`.
    """
    
    x_choice_extended = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', 'receivables', 'net_ppe', 'capex', \
                        'intangible_assets', 'lt_debt', 'policy_emissions_score', 'target_emissions_score']
    x_choice_core = ['mcap', 'revenue', 'capex', 'intangible_assets', 'lt_debt']
    
    # Drop missings if any of the extended choice set is missing
    missings_to_remove_cols = y_choice + ['s1_and_s2_co2e', 's1_and_s2_co2e_intensity'] + x_choice_extended
    len_1 = len(df)
    df.dropna(subset=missings_to_remove_cols, inplace=True)
    len_2 = len(df)
    if verbose:
        print(f'Dropped {len_1 - len_2} missing observations')
    
    # Drop zeros for our core variables as we will most likely log them
    zero_cols_to_check = x_choice_core + ['s1_and_s2_co2e', 's1_and_s2_co2e_intensity'] + y_choice
    df_filtered = df[(df[zero_cols_to_check] > 0).all(axis=1)]
    len_3 = len(df_filtered)
    if verbose:
        print(f'Dropped {len_2 - len_3} non-positive observations')
    return df_filtered
   

def load_and_create_peer_groups(df, y_choice_str, sectors, min_firms_per_fold, n_folds, seed):
    """
    Generates peer groups for firms based on specified sector features and creates stratified folds for cross-validation.
    Additionally, one-hot encodes the generated peer groups and appends these as new columns to the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        y_choice_str (str): The target variable name as a string, used for creating peer groups.
        sectors (list): A list of sector features (columns) used to group the firms into peer groups.
        min_firms_per_fold (int): The minimum number of firms required per fold when creating peer groups.
        n_folds (int): The number of folds to use for cross-validation.
        seed (int): A seed for random number generation to ensure reproducibility.

    Returns:
        pandas.DataFrame: The original DataFrame with additional one-hot encoded columns representing the peer groups.
    """
    df = get_peer_groups_and_folds(df, y_choice_str, sector_features=sectors, minimum_firms_per_fold=min_firms_per_fold, k_folds=n_folds, seed_num=seed, verbose=True)
    df = pd.concat([df, pd.get_dummies(df['peer_group'], prefix='peer_group', drop_first=True)], axis=1)
    return df

    
def prepare_base_dataframe(df, y_choice, x_choice_str, year_lower, year_upper, observation_filter, k_folds, min_firms_per_fold, seed_num):
    """
    Prepares the base DataFrame for modeling by performing various preprocessing steps such as subsetting the 
    data to a specific date range, removing missing and non-positive values, applying observation filters, 
    and generating peer groups. The function also returns the feature variables (X) based on the target 
    variable (Y) and a specified feature set.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the raw data.
        y_choice (list): A list with a single target variable for the model, representing either absolute or 
            intensity measures of emissions. Must be one of the following:
            - Absolute: ['s3_upstream', 's3_downstream', 's3_cat_total', 's1_and_s2_co2e', 's3_purchased_goods_cat1']
            - Intensity: ['s3_upstream_intensity', 's3_downstream_intensity', 's3_cat_total_intensity', 's1_and_s2_co2e_intensity', 's3_purchased_goods_cat1_intensity'].
        x_choice_str (str): Specifies which set of feature variables to use. Must be one of the following:
            - 'extended': Returns a comprehensive set of features, including additional variables like ebit and assets.
            - 'core': Returns a core set of features, more limited than the 'extended' set.
            - 'limited': Returns the most limited set of features, typically for baseline models.
        year_lower (int): The lower bound of the year range (inclusive) for filtering the data.
        year_upper (int): The upper bound of the year range (inclusive) for filtering the data.
        observation_filter (str or None): A flag indicating whether to apply a specific filter to the observations. 
            - If None, don't apply any filter and use the observation_filter_str as "all" (to imply all data is used).
            - If "covid", exclude all observations in the year 2021 and set observation_filter_str to "ExCovid".
            - If "material", apply the materiality filter and set observation_filter_str to "material".
        k_folds (int): The number of folds to use when creating peer groups for cross-validation.
        min_firms_per_fold (int): The minimum number of firms required per fold when creating peer groups.
        seed_num (int): A seed for random number generation to ensure reproducibility.

    Returns:
        tuple: A tuple containing the following elements:
            - pandas.DataFrame: The preprocessed DataFrame ready for modeling.
            - list: A list of feature variables (X) corresponding to the specified target variable (Y) 
              and the chosen feature set.
            - list: A list of variables requiring scaling.

    Raises:
        ValueError: If `y_choice` is not in the predefined set of acceptable target variables.
        ValueError: If `x_choice_str` is not one of 'extended', 'core', or 'limited'.
        ValueError: If `observation_filter` is not one of: None, "covid", or "material".
    """
    
    # Define x_choice variables inside the function to avoid mutation across iterations
    y_choice_absolute = ['s3_upstream', 's3_downstream', 's3_cat_total', 's1_and_s2_co2e', 's3_purchased_goods_cat1', 's3_business_travel_cat6']
    y_choice_intensity = [x +'_intensity' for x in y_choice_absolute]
    y_choice_set = y_choice_absolute + y_choice_intensity
    x_choice_extended = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', 'receivables', 'net_ppe', 'capex', \
                        'intangible_assets', 'lt_debt', 'policy_emissions_score', 'target_emissions_score']
    x_choice_core = ['mcap', 'revenue', 'capex', 'intangible_assets', 'lt_debt']
    x_choice_limited = ['mcap', 'revenue']

    # Validate y_choice
    if y_choice[0] not in y_choice_set:
        raise ValueError(f"y_choice must be in {y_choice_set}")
    
    # Adjust x_choice based on the feature set selection (extended, core, limited)
    y_choice_type = 'absolute'
    if y_choice[0] in y_choice_intensity:
        y_choice_type = 'intensity'
    
    if y_choice_type == 'absolute':
        if y_choice[0] != 's1_and_s2_co2e':
            x_choice_extended.append('s1_and_s2_co2e')
            x_choice_core.append('s1_and_s2_co2e')
            x_choice_limited.append('s1_and_s2_co2e')
    else:
        if y_choice[0] != 's1_and_s2_co2e_intensity':
            x_choice_extended.append('s1_and_s2_co2e_intensity')
            x_choice_core.append('s1_and_s2_co2e_intensity')
            x_choice_limited.append('s1_and_s2_co2e_intensity')

    if x_choice_str == 'extended':
        x_choice  = x_choice_extended
    elif x_choice_str == 'core':
        x_choice = x_choice_core
    elif x_choice_str == 'limited':
        x_choice = x_choice_limited
    else:
        raise ValueError('x_choice_str must be either "extended", "core", or "limited"')

    # First, apply the observation filter
    if observation_filter is None:
        observation_filter_str = 'all'
    elif observation_filter == "covid":
        df = df[df['year'] != 2021] # assume for now that covid impact is contained to 2021 reporting year
        observation_filter_str = 'ExCovid'
    elif observation_filter == "material":
        if str(y_choice[0]).startswith('s3_'):
            df = df[get_material_s3_category_reporting(df)]
        else:
            print("As the target variable doesn't relate to S3, materiality filter doesn't apply")
        observation_filter_str = 'material'
    else:
        raise ValueError('observation_filter must be one of: None, "covid", or "material"')

    # Second, subset to the required years
    df = subset_to_date_range(df, year_lower=year_lower, year_upper=year_upper)

    # Third, remove any missings and zero values which would make no sense
    df = remove_missing_and_non_positive_values(df, y_choice)

    # Fourth, get peer groups
    sectors = ['econ_sector', 'business_sector', 'industry_group_sector', 'industry_sector', 'activity_sector']
    df = load_and_create_peer_groups(df, y_choice_str=y_choice[0], sectors=sectors, min_firms_per_fold=min_firms_per_fold, n_folds=k_folds, seed=seed_num)
    
    # Return the dataframe, the choice variables, and the variables requiring scaling
    FY_dummies = [x for x in df.columns if str(x).startswith('FY_')]
    econ_type_dummies = [x for x in df.columns if 'cc_classification_' in str(x)]
    peer_dummies = [x for x in df.columns if 'peer_group_' in str(x)]
    dummies = FY_dummies + econ_type_dummies + peer_dummies
    x_choice_vars = x_choice + dummies
    return df, x_choice_vars, x_choice, observation_filter_str


def get_train_test_indices(df):
    """
    Generates train and test indices based on a random fold selection, and 
    prepares custom folds for cross-validation.

    This function randomly selects one of the folds in the `df['fold']` column 
    to be the test fold, and returns the indices for training and testing sets 
    accordingly. Additionally, it generates custom folds for cross-validation 
    using the remaining data.

    Args:
        df (pandas.DataFrame): The input DataFrame that must contain a 'fold' column 
        indicating the fold assignment for each row.

    Returns:
        tuple: A tuple containing:
            - train_inds (numpy.ndarray): The indices of the training data.
            - test_inds (numpy.ndarray): The indices of the test data.
            - custom_folds_for_training_data (list of tuples): A list of tuples, 
              where each tuple contains two lists:
              - train_indices (list): Indices for training in the fold.
              - test_indices (list): Indices for testing in the fold.

    Raises:
        ValueError: If `df['fold']` does not contain unique fold values.
    """
    n_folds = len(df['fold'].unique())
    
    # Pick a random fold for our test fold
    random_test_fold = np.random.randint(1, n_folds + 1)
    print(f'\nThe random test fold is {random_test_fold}')
    
    # Get the train and test indices over the whole dataframe
    train_inds = np.array(df.index[df['fold'] != random_test_fold])
    test_inds = np.array(df.index[df['fold'] == random_test_fold])
    
    # For the training data, get the indices for n-fold minus 1 cross-validation
    custom_folds_for_training_data = []
    training_data = df.iloc[train_inds]
    training_data.reset_index(inplace=True)
    for i in list(training_data['fold'].unique()):
        train_indices = training_data.index[training_data['fold'] != i].tolist()
        test_indices = training_data.index[training_data['fold'] == i].tolist() 
        custom_folds_for_training_data.append((train_indices, test_indices))
        
    return train_inds, test_inds, custom_folds_for_training_data


def get_train_test_split(df, training_indices, test_indices, x_choice, y_choice, id_choice):
    """
    Splits the DataFrame into training and test sets based on provided indices, and 
    returns both the full DataFrame splits and the individual numpy arrays for features, 
    target variables, and IDs.

    This function separates the input DataFrame into training and test datasets 
    according to the specified indices. It then returns these datasets as both 
    complete DataFrames and as individual numpy arrays for features (X), target 
    variables (y), and ID columns.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the full dataset.
        training_indices (array-like): The indices used to select the training data from `df`.
        test_indices (array-like): The indices used to select the test data from `df`.
        x_choice (list): A list of column names corresponding to the feature variables.
        y_choice (list): A list of column names corresponding to the target variables.
        id_choice (list): A list of column names corresponding to the ID variables.

    Returns:
        tuple: A tuple containing:
            - data_train (pandas.DataFrame): The DataFrame containing the training data.
            - data_test (pandas.DataFrame): The DataFrame containing the test data.
            - X_train (numpy.ndarray): The feature array for the training data.
            - X_test (numpy.ndarray): The feature array for the test data.
            - y_train (numpy.ndarray): The target variable array for the training data.
            - y_test (numpy.ndarray): The target variable array for the test data.
            - ID_train (numpy.ndarray): The ID array for the training data.
            - ID_test (numpy.ndarray): The ID array for the test data.
    """
    
    X, y, group = df[x_choice].values, df[y_choice].values, df[id_choice].values
    X_train, X_test, y_train, y_test, ID_train, ID_test = X[training_indices], X[test_indices], y[training_indices], y[test_indices], group[training_indices], group[test_indices]
    print("\nDataset train has {} entries and {} features".format(*X_train.shape))
    print("Dataset test has {} entries and {} features".format(*X_test.shape))
    
    data_train = pd.concat([pd.DataFrame(ID_train, columns=id_choice),
                            pd.DataFrame(X_train, columns=x_choice),
                            pd.DataFrame(y_train, columns=y_choice)], axis=1)

    data_test = pd.concat([pd.DataFrame(ID_test, columns=id_choice),
                        pd.DataFrame(X_test, columns=x_choice),
                        pd.DataFrame(y_test, columns=y_choice)], axis=1)
    
    return data_train, data_test, X_train, X_test, y_train, y_test, ID_train, ID_test


def scale_data(training_data, test_data, x_choice_str, scaler_choice, y_choice, x_choice, id_choice, ID_train, ID_test, vars_to_scale):
    """
    Scales the data based on the specified scaling method and choice of features.

    Args:
        training_data (pandas.DataFrame): The training data containing features and target variables.
        test_data (pandas.DataFrame): The test data containing features and target variables.
        x_choice_str (str): Specifies the set of features to use. Must be one of 'extended', 'core', or 'limited'.
        scaler_choice (object): The scaler object, such as RobustScaler() or MinMaxScaler(), to be used if 'extended' is chosen.
        y_choice (list): List of target variable(s) column names.
        x_choice (list): List of feature variable column names.
        id_choice (list): List of ID column names.
        ID_train (numpy.ndarray): ID array for the training data.
        ID_test (numpy.ndarray): ID array for the test data.
        vars_to_scale (list): List of variable column names to scale.

    Returns:
        tuple: A tuple containing:
            - data_train_scaled (pandas.DataFrame): Scaled training data.
            - data_test_scaled (pandas.DataFrame): Scaled test data.
            - X_train_scaled (numpy.ndarray): Scaled feature array for the training data.
            - X_test_scaled (numpy.ndarray): Scaled feature array for the test data.
            - y_train_scaled (numpy.ndarray): Scaled target array for the training data.
            - y_test_scaled (numpy.ndarray): Scaled target array for the test data.
            - scaler_fit (object or None): The fitted scaler object, or None if log transformation is used.

    Raises:
        ValueError: If `x_choice_str` is 'extended' and `scaler_choice` is not RobustScaler() or MinMaxScaler().
    """
    
    if x_choice_str == 'extended':
        # Validate that scaler_choice is either RobustScaler or MinMaxScaler
        if not isinstance(scaler_choice, (RobustScaler, MinMaxScaler)):
            raise ValueError("For 'extended' x_choice_str, scaler_choice must be either RobustScaler() or MinMaxScaler().")

    training_scaled = training_data.copy()
    test_scaled = test_data.copy()

    if x_choice_str == 'extended':
        # Apply robust or minmax scaling
        scaler_fit = scaler_choice.fit(training_scaled[vars_to_scale])  
        training_scaled[vars_to_scale] = scaler_fit.transform(training_scaled[vars_to_scale])
        test_scaled[vars_to_scale] = scaler_fit.transform(test_scaled[vars_to_scale])  # scale to train fit
    else:
        # Apply log transformation
        scaler_fit = None  # No scaler is used for log transformation
        for var in vars_to_scale:
            training_scaled[var] = np.log(training_scaled[var])
            test_scaled[var] = np.log(test_scaled[var])

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = \
        training_scaled[x_choice].values, test_scaled[x_choice].values, \
        training_scaled[y_choice].values, test_scaled[y_choice].values
        
    data_train_scaled = pd.concat([pd.DataFrame(ID_train, columns=id_choice),
                            pd.DataFrame(X_train_scaled, columns=x_choice),
                            pd.DataFrame(y_train_scaled, columns=y_choice)], axis=1)

    data_test_scaled = pd.concat([pd.DataFrame(ID_test, columns=id_choice),
                            pd.DataFrame(X_test_scaled, columns=x_choice),
                            pd.DataFrame(y_test_scaled, columns=y_choice)], axis=1)
    
    return data_train_scaled, data_test_scaled, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_fit


def unscale_predictions(y_pred_scaled, X_scaled_data, x_choice_vars, scaler, x_choice_str, x_to_unscale):
    """
    Unscales or unlogs the predicted values based on the scaling or transformation applied during preprocessing.

    Args:
        y_pred_scaled (numpy.ndarray): The scaled/logged predicted values.
        X_scaled_data (numpy.ndarray): The scaled/logged feature data.
        x_choice_vars (list): The list of feature variable names.
        scaler (object or None): The scaler object used during preprocessing, or None if log transformation was used.
        x_choice_str (str): Specifies the set of features used. Must be one of 'extended', 'core', or 'limited'.
        x_to_unscale (list): Which X variables to transform back into original scale.

    Returns:
        pandas.Series: The unscaled or unlogged predicted values.
    """
    x_vars_scaled = x_to_unscale
    if x_choice_str == 'extended':
        # Apply inverse scaling using the scaler
        X_scaled_transformed = pd.DataFrame(X_scaled_data, columns=x_choice_vars)[x_vars_scaled]
        data_scaled = np.column_stack((X_scaled_transformed, y_pred_scaled))
        y_pred_unscaled = pd.Series(scaler.inverse_transform(data_scaled)[:, -1])
    else:
        # Apply inverse log transformation
        y_pred_unscaled = np.exp(y_pred_scaled)
    
    return y_pred_unscaled


def model_tune(model, param_grid, X_train, y_train, custom_folds, save_path, model_intials, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str):
    """
    Tunes a given model using GridSearchCV, saves the best model, and prints the best estimator.

    Args:
        model (sklearn estimator): The machine learning model to be tuned.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        X_train (numpy.ndarray or pandas.DataFrame): The training input samples.
        y_train (numpy.ndarray or pandas.Series): The target values for training.
        custom_folds (list of tuples): Predefined train/test split indices for cross-validation.
        save_path (str): Directory path where the tuned model will be saved.
        model_initials (str): Initials or short name of the model (e.g., 'RF' for RandomForest).
        y_choice (list): The target variable name (wrapped in a list).
        x_choice_str (str): Specifies the feature set used ('extended', 'core', or 'limited').
        observation_filter_str (str): Flags whether we filter any observations out.
        year_lower (int): The lower bound of the year range used for training.
        year_upper (int): The upper bound of the year range used for training.
        scaler_str (str): String representing the scaler used (e.g., 'robust', 'minmax').

    Returns:
        None
    """
    search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=custom_folds, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train.ravel())
    model_name = model_intials + '_' + y_choice[0] + '_' + x_choice_str + '_' + observation_filter_str + '_' + str(year_lower) + '_' + str(year_upper) + '_' + scaler_str
    # Save the best model
    print('Best model is {}'.format(search.best_estimator_))
    
    with open(f'{save_path}/{model_name}.pickle', 'wb') as handle:
        pickle.dump(search.best_estimator_, handle)


def run_all_model_tuning(X_train, y_train, custom_folds, save_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, seed_choice=42):
    """
    Tunes several machine learning models using predefined hyperparameter grids, saves the best models, and prints the best estimators.

    Args:
        X_train (numpy.ndarray or pandas.DataFrame): The training input samples.
        y_train (numpy.ndarray or pandas.Series): The target values for training.
        custom_folds (list of tuples): Predefined train/test split indices for cross-validation.
        save_path (str): Directory path where the tuned models will be saved.
        y_choice (list): The target variable name (wrapped in a list).
        x_choice_str (str): Specifies the feature set used ('extended', 'core', or 'limited').
        observation_filter_str (str): Flags whether we filter any observations out.
        year_lower (int): The lower bound of the year range used for training.
        year_upper (int): The upper bound of the year range used for training.
        scaler_str (str): String representing the scaler used (e.g., 'robust', 'minmax').
        seed_choice (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        None
    """
    models_and_grids = {
        'QR': {
            'model_class': QuantileRegressor(),
            'param_grid': {'quantile': [0.1, 0.25, 0.5, 0.75, 0.9], 'alpha': [0.01, 0.1, 0.5, 1.0, 10.0], 'solver': ['highs-ds', 'highs-ipm', 'highs'], 'fit_intercept': [True, False]}
        },
        'EL': {
            'model_class': ElasticNet(max_iter=10000, random_state=seed_choice),
            'param_grid': {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0], 'l1_ratio': np.arange(0, 1, 0.05)}
        },
        'MLP': {
            'model_class': MLPRegressor(random_state=seed_choice),
            'param_grid': {
                'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50)],
                'activation': ['tanh', 'relu'],
                'alpha': [1e-4, 1e-3, 1e-1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': np.arange(0.1, 1.1, 0.2),
                'max_iter': list(range(1000, 2500, 500)),
                'early_stopping': [True, False]
            }
        },
        'KNN': {
            'model_class': KNeighborsRegressor(),
            'param_grid': {'n_neighbors': np.arange(5, 505, 5), 'algorithm': ["auto", "ball_tree", "kd_tree", "brute"], 'weights': ['uniform', 'distance'], 'metric': ["minkowski", "euclidean", "manhattan"], 'p': np.arange(3, 5)}
        },
        'RF': {
            'model_class': RandomForestRegressor(criterion='absolute_error', random_state=seed_choice),
            'param_grid': {
                'n_estimators': [int(x) for x in np.arange(start=100, stop=400, step=100)],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [2, 3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 8, 10],
                'bootstrap': [True, False]
            }
        }
    }

    for model_initials, model_info in models_and_grids.items():
        model_class = model_info['model_class']
        param_grid = model_info['param_grid']

        # Call model_tune for each model
        model_tune(
            model=model_class,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            custom_folds=custom_folds,
            save_path=save_path,
            model_intials=model_initials,
            y_choice=y_choice,
            x_choice_str=x_choice_str,
            observation_filter_str=observation_filter_str,
            year_lower=year_lower,
            year_upper=year_upper,
            scaler_str=scaler_str
        )

    # XGBoost model tuning (special case)
    dtrain_scaled = xgb.DMatrix(X_train, label=y_train)
    
    params_xgb = {
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': .3,
        'lambda': 0,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'mae'
    }

    param_grid_xgb = {
        'max_depth': list(range(6, 12)),
        'min_child_weight': list(range(1, 9)),
        'eta': [0.00001, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'lambda': [0.00001, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    best_params = {}  
    best_boost_rounds = 0  

    for parameter, values in param_grid_xgb.items():
        
        min_eval_score = float('inf')
        
        for val_to_try in param_grid_xgb[parameter]:
            # Update the parameter value to cross-validate
            params_xgb[parameter] = val_to_try
            
            cv_results = xgb.cv(params=params_xgb, dtrain=dtrain_scaled, num_boost_round=1000, folds=custom_folds, early_stopping_rounds=50, seed=seed_choice)
            
            # Update best MAE
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            if mean_mae < min_eval_score:
                min_eval_score = mean_mae
                best_params[parameter] = val_to_try
                best_boost_rounds = boost_rounds
            
        params_xgb.update(best_params)
        
    model_name = 'XGB' + '_' + y_choice[0] + '_' + x_choice_str + '_' + observation_filter_str + '_' + str(year_lower) + '_' + str(year_upper) + '_' + scaler_str
    print(f'Best parameters for XGB are: {best_params} with {best_boost_rounds} boost rounds')

    with open(f'{save_path}/{model_name}' + '.pickle', 'wb') as handle:
        pickle.dump((best_params, best_boost_rounds), handle)


def prepare_data_pipeline(data_file_path, y_choice, x_choice_str, scaler_choice, year_lower, year_upper, observation_filter, 
                          min_firms_per_fold, n_folds, seed_choice):
    """
    Prepares the data for model training, including loading, filtering, scaling, and splitting.

    Args:
        data_file_path (str): The path to the dataset CSV file.
        y_choice (list): The target variable name (wrapped in a list).
        x_choice_str (str): Specifies the feature set used ('extended', 'core', or 'limited').
        scaler_choice (object): The scaler object, such as RobustScaler() or MinMaxScaler(), or None for log transformation.
        year_lower (int): The lower bound of the year range used for training.
        year_upper (int): The upper bound of the year range used for training.
        observation_filter (str): A flag to indicate which observations to filter before modelling. (None, "covid", "material")
        min_firms_per_fold (int): The minimum number of firms required per fold when creating peer groups.
        n_folds (int): The number of folds for cross-validation.
        seed_choice (int): Random seed for reproducibility.

    Returns:
        tuple: Contains the prepared training and test datasets, along with other relevant data.
    """
    # Determine the appropriate scaler string
    if x_choice_str in ['core', 'limited']:
        scaler_str = 'log'
    elif isinstance(scaler_choice, RobustScaler):
        scaler_str = 'robust'
    elif isinstance(scaler_choice, MinMaxScaler):
        scaler_str = 'minmax'
    else:
        raise ValueError('scaler_choice must be either RobustScaler(), MinMaxScaler(), or None')

    # Load the data
    df = pd.read_csv(data_file_path)
    
    # Prepare the base DataFrame
    df, x_choice, x_scale, observation_filter_str = prepare_base_dataframe(df, y_choice, x_choice_str, year_lower, year_upper, observation_filter, 
                                                   k_folds=n_folds, min_firms_per_fold=min_firms_per_fold, seed_num=seed_choice)
    
    # Get train and test indices
    train_inds, test_inds, custom_folds_for_training_data = get_train_test_indices(df)

    # Split the data
    data_train, data_test, X_train, X_test, y_train, y_test, ID_train, ID_test = \
        get_train_test_split(df, training_indices=train_inds, test_indices=test_inds, x_choice=x_choice, y_choice=y_choice, id_choice=['instrument'])

    # Scale the data
    data_train_scaled, data_test_scaled, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_fitted = \
        scale_data(training_data=data_train, test_data=data_test, x_choice_str=x_choice_str, scaler_choice=scaler_choice, 
                   y_choice=y_choice, x_choice=x_choice, id_choice=['instrument'], ID_train=ID_train, ID_test=ID_test, vars_to_scale=x_scale + y_choice)

    return data_train_scaled, data_test_scaled, data_test, X_train_scaled, X_test_scaled, x_choice, x_scale, y_train_scaled, y_test_scaled, y_test, custom_folds_for_training_data, scaler_str, observation_filter_str


def get_results(y_test_data, y_pred_unscaled_data):
    
    test_mae = np.round(mean_absolute_error(y_test_data, y_pred_unscaled_data), 0)
    test_median_AE = np.round(median_absolute_error(y_test_data, y_pred_unscaled_data), 0)
    test_mape = np.round(mean_absolute_percentage_error(y_test_data, y_pred_unscaled_data), 2)
    test_r2 = np.round(r2_score(y_test_data, y_pred_unscaled_data), 2)
    
    return {'MAE': test_mae, 'Median AE': test_median_AE, 'MAPE': test_mape, 'R^2': test_r2}


def get_model_path(save_path, model_intials, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str):
    model_path = save_path + model_intials + '_' + y_choice[0] + '_' + x_choice_str + '_' + observation_filter_str + '_' + str(year_lower) + '_' + str(year_upper) + '_' + scaler_str + '.pickle'
    return model_path


def get_model_results(model_initials, save_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                      X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled, dtrain_scaled=None, dtest_scaled=None):
    """
    Loads a saved model, makes predictions on the test set, unscales the predictions, 
    and returns the evaluation results. Handles both scikit-learn models and XGBoost models.

    Args:
        model_initials (str): The initials of the model (e.g., 'QR' for Quantile Regressor, 'XGB' for XGBoost).
        save_path (str): The directory where the model is saved.
        y_choice (list): The target variable name (wrapped in a list).
        x_choice_str (str): Specifies the feature set used ('extended', 'core', or 'limited').
        observation_filter_str (str): Whether we filter any data or not. (Do we keep "all", "ExCovid", or "material" obs?)
        year_lower (int): The lower bound of the year range used for training.
        year_upper (int): The upper bound of the year range used for training.
        scaler_str (str): String representing the scaler used (e.g., 'robust', 'minmax', 'log').
        X_train_scaled (numpy.ndarray): Scaled training feature data.
        y_train_scaled (numpy.ndarray): Scaled training target data.
        X_test_scaled (numpy.ndarray): Scaled test feature data.
        y_test (numpy.ndarray): True test target data.
        x_choice (list): List of feature variable names.
        scaler_choice (object): The scaler used during preprocessing (e.g., RobustScaler(), MinMaxScaler()).
        x_scaled (list): List of variables to be unscaled.
        dtrain_scaled (xgb.DMatrix, optional): DMatrix for XGBoost training data.
        dtest_scaled (xgb.DMatrix, optional): DMatrix for XGBoost test data.

    Returns:
        tuple: A dictionary containing the evaluation results, and the unscaled predictions of the target variable.
    """
    if model_initials == 'XGB':
        # Load the XGBoost model parameters and number of boost rounds
        model_path = get_model_path(save_path, model_initials, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str)
        XGB_params, optimal_boost_rounds = pickle.load(open(model_path, 'rb'))

        # Train the XGBoost model
        model = xgb.train(params=XGB_params, dtrain=dtrain_scaled, num_boost_round=optimal_boost_rounds)

        # Make predictions on the test set
        y_pred_scaled = model.predict(dtest_scaled)

        # Unscale the predictions
        y_pred_unscaled = unscale_predictions(y_pred_scaled=y_pred_scaled, X_scaled_data=X_test_scaled, x_choice_vars=x_choice,
                                              scaler=scaler_choice, x_choice_str=x_choice_str, x_to_unscale=x_scaled)
    else:
        # Load the scikit-learn model
        model_path = get_model_path(save_path, model_initials, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str)
        model = pickle.load(open(model_path, 'rb'))

        # Fit the model on the training data (if not already fitted)
        model.fit(X_train_scaled, y_train_scaled.ravel())

        # Make predictions on the test set
        y_pred_scaled = model.predict(X_test_scaled)

        # Unscale the predictions
        y_pred_unscaled = unscale_predictions(y_pred_scaled=y_pred_scaled, X_scaled_data=X_test_scaled, x_choice_vars=x_choice,
                                              scaler=scaler_choice, x_choice_str=x_choice_str, x_to_unscale=x_scaled)

    # Get evaluation results
    results = get_results(y_test, y_pred_unscaled)
    
    return results, y_pred_unscaled
