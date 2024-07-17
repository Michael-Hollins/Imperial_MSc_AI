
import numpy as np
import pandas as pd

def fill_missings_and_consolidate(df, year_lower, year_upper):
    """
    Processes the input DataFrame by sorting, filling missing values, and subsetting based on the given year range.
    For each column, it keeps the last non-missing value in the [firm_name, year] combination.

    This function performs the following steps:
    1. Sorts the DataFrame by 'firm_name' and 'year'.
    2. For each group of 'firm_name' and 'year', it forward-fills and then backward-fills missing values, 
       ensuring that the last non-missing value is kept.
    3. Subsets the DataFrame to include only rows where 'year' is within the specified range [year_lower, year_upper].

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns 'firm_name', 'year', and 'co2e_method'.
        year_lower (int): The lower bound of the year range for subsetting.
        year_upper (int): The upper bound of the year range for subsetting.

    Returns:
        pandas.DataFrame: The processed DataFrame with missing values filled and subset based on the specified year range.
    """
    # Step 1: Sort the DataFrame by 'firm_name' and 'year'
    df = df.sort_values(by=['firm_name', 'year']).reset_index(drop=True)
    df['co2e_method'] = df['co2e_method'].apply(lambda x: x if x == 'Reported' else np.nan)
    
    def get_last_non_missing(group):
        return group.ffill().bfill().iloc[-1]
    
    # Step 2: Apply the helper function to each group of 'firm_name' and 'year'
    df = df.groupby(['firm_name', 'year']).apply(get_last_non_missing).reset_index(drop=True)
    
    # Step 3: Subset the DataFrame to include only rows within the specified year range
    df = df[(df['year'] >= year_lower) & (df['year'] <= year_upper)]
    
    return df


def process_missings_from_load(df, year_lower, year_upper, verbose=False):
    """
    Processes the input DataFrame by standardising missing values, filling and consolidating data on a yearly and firm basis, 
    and dropping rows where all specified historical values are missing.

    Args:
        df (pandas.DataFrame): The input DataFrame containing data to be processed. Must include 'revenue' and 's3_investments_cat15' columns for historical fields, and 'co2e_method' for consolidation.
        year_lower (int): The lower bound of the year range for subsetting in the consolidation step.
        year_upper (int): The upper bound of the year range for subsetting in the consolidation step.
        verbose (bool, optional): If True, prints the number of observations dropped and remaining after processing. Default is False.

    Returns:
        pandas.DataFrame: The processed DataFrame with missing values standardised, filled, consolidated, and with rows dropped where all historical values are missing.
    """
    df = df.applymap(lambda x: np.nan if pd.isna(x) else x)
    df.replace('', np.nan, inplace=True)
    
    # fill out missings and consolidate on a year-firm basis
    df = fill_missings_and_consolidate(df, year_lower=year_lower, year_upper=year_upper)
    
    # drop rows where everything is missing
    start_col = df.columns.get_loc('revenue')
    end_col = df.columns.get_loc('s3_investments_cat15')
    historical_fields = df.columns[start_col:end_col + 1].tolist()
    historical_fields.remove('co2e_method')
    n1 = len(df) 
    data_no_all_missing = df.dropna(subset=historical_fields, how='all')   
    n2 = len(data_no_all_missing)
    
    if verbose:    
        print(f'Data load complete. {n1-n2} observations dropped because all historical values NA. {n2} observations remaining.')
    
    return data_no_all_missing


def keep_last_observation_per_year(df):
    """
    Retains the last observation for each firm and year combination from the input DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the columns 'firm_name', 'year', and 'date_val'.

    Returns:
        pandas.DataFrame: A DataFrame with the last observation for each 'firm_name' and 'year' combination, without the 'date_val' column.
    """
    df = df.sort_values(by=['firm_name', 'year', 'date_val'])
    result = df.groupby(['firm_name', 'year']).last().reset_index()
    result = result.drop(columns=['date_val'])
    return result



    
    
    

    
    
     