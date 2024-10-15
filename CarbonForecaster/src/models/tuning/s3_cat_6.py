# Category 6 tuning
######################################################
################### TUNING SCRIPT ####################
######################################################
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import *
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import itertools

def generate_param_combinations():
    # Define the base lists for combinations
    y_choice_list = [['s3_business_travel_cat6'], ['s3_business_travel_cat6_intensity']]
    x_choice_str_list = ['extended', 'core', 'limited']
    observation_filter_list = [None, 'covid']
    # Initialize an empty list to store all parameter combinations
    param_combinations = []

    # Loop over all combinations of y_choice, x_choice_str, and observation_filter
    for y_choice, x_choice_str, observation_filter in itertools.product(y_choice_list, x_choice_str_list, observation_filter_list):
        # Set scaler_choice based on x_choice_str
        if x_choice_str == 'extended':
            scaler_choice = RobustScaler()
        else:
            scaler_choice = None
        
        # Set n_folds and min_firms_per_fold based on observation_filter
        if observation_filter == 'material':
            n_folds = 7
            min_firms_per_fold = 2
        else:
            n_folds = 8
            min_firms_per_fold = 3
        
        # Create a dictionary of the parameters for this combination
        param_combination = {
            'y_choice': y_choice,
            'x_choice_str': x_choice_str,
            'scaler_choice': scaler_choice,
            'observation_filter': observation_filter,
            'n_folds': n_folds,
            'min_firms_per_fold': min_firms_per_fold
        }

        # Append this combination to the list
        param_combinations.append(param_combination)

    return param_combinations

param_combination = generate_param_combinations()


# Other parameters
year_lower = 2016
year_upper = 2022
seed_choice = 42
np.random.seed(seed_choice)

for params in param_combination:
    # Data preparation
    data_train_scaled, data_test_scaled, data_test, X_train_scaled, X_test_scaled, x_choice, x_scaled, y_train_scaled, y_test_scaled, y_test, custom_folds_for_training_data, scaler_str, observation_filter_str = \
        prepare_data_pipeline(
            data_file_path='data/ftse_world_allcap_clean.csv', 
            y_choice=params['y_choice'], 
            x_choice_str=params['x_choice_str'], 
            scaler_choice=params['scaler_choice'], 
            year_lower=year_lower, 
            year_upper=year_upper, 
            observation_filter=params['observation_filter'], 
            min_firms_per_fold=params['min_firms_per_fold'], 
            n_folds=params['n_folds'], 
            seed_choice=seed_choice
        )
        
        # Hyperparameter tuning
    run_all_model_tuning(
        X_train=X_train_scaled, 
        y_train=y_train_scaled, 
        custom_folds=custom_folds_for_training_data, 
        save_path='src/models/tuned', 
        y_choice=params['y_choice'], 
        x_choice_str=params['x_choice_str'], 
        observation_filter_str=observation_filter_str, 
        year_lower=year_lower, 
        year_upper=year_upper, 
        scaler_str=scaler_str, 
        seed_choice=seed_choice
    )