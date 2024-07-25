
import numpy as np
import pandas as pd
from load_raw_data import load_from_excel
from load_raw_data import col_mapping
import pickle

def match_absolute_and_relative_financial_years(excel_file_path, api_file_path):
    """
    Matches absolute and relative financial years by aligning revenues from two data sources.

    This function reads data from an Excel file and a pickled DataFrame, sorts and filters the data,
    and then merges the two DataFrames based on matching 'instrument' and 'revenue' values. It returns
    a DataFrame with 'instrument', 'absolute_financial_year', and 'relative_financial_year'.
    
    This is needed because some firms have the previous financial year as e.g. 2023 and others have it as 2024.

    Args:
        excel_file_path (str): The file path to the Excel file containing 'static_data' and 'historical_data'.
        api_file_path (str): The file path to the pickled DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing 'instrument', 'absolute_financial_year', and 'relative_financial_year'
                      for matching 'instrument' and 'revenue' values from the two data sources. It returns the index
                      for only where 'revenue' isn't missing.
    """
    df_excel = load_from_excel(excel_file_path, 'static_data', 'historical_data')
    df_excel['absolute_financial_year_numeric'] = df_excel['absolute_financial_year'].str.extract('(\d+)').astype(int)
    df_excel = df_excel.groupby('instrument').apply(lambda x: x.sort_values(by='absolute_financial_year_numeric', ascending=True)).reset_index(drop=True)
    df_excel = df_excel.drop(columns='absolute_financial_year_numeric')
    df_excel = df_excel[['instrument', 'absolute_financial_year', 'revenue']].dropna()
    
    df_api = pd.read_pickle(api_file_path)
    df_api.sort_values(by='instrument').reset_index(drop=True)
    df_api = df_api[['instrument', 'relative_financial_year', 'revenue']].dropna()
    
    firms_by_year = df_excel.merge(df_api, on=['instrument', 'revenue'])
    firms_by_year['year'] = firms_by_year['absolute_financial_year'].str.replace('FY', '').astype(int)
    return firms_by_year[['instrument', 'absolute_financial_year', 'relative_financial_year', 'year']]


def standardise_missing_values(df):
    """
    Standardises missing values in a DataFrame.

    This function replaces any form of missing values in the DataFrame with `np.nan`.
    It standardizes missing values represented as empty strings or other NA types.

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized missing values.
    """
    df = df.applymap(lambda x: np.nan if pd.isna(x) else x)
    df.replace('', np.nan, inplace=True)
    return df


def mask_non_reported_co2e(df):
    """
    Masks CO2e values where the co2e_method is not 'Reported'.

    This function first sets the co2e_method to either 'Reported' or missing. Next, it modifies the emissions columns 
    to NaN for rows where the 'co2e_method' is not 'Reported'.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the columns 'co2e_method' and emissions data.

    Returns:
        pandas.DataFrame: The DataFrame with masked CO2e values where 'co2e_method' is not 'Reported'.
    """
    df['co2e_method'] = df['co2e_method'].apply(lambda x: x if x == 'Reported' else np.nan)
    df.loc[df['co2e_method'] != 'Reported', ['s1_co2e', 's2_co2e', 's3_co2e',
                                             'policy_emissions_score', 'target_emissions_score', 'emissions_trading_score']] = np.nan
    return df


def consolidation_is_possible(df, group_cols, historical_cols):
    """
    Checks that the raw data doesn't contradict itself by having different values per group.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.
        group_cols (list): List of columns to group by (e.g., ['firm_name', 'year']).
        historical_cols (list): List of historical columns to check for consolidation.

    Returns:
        tuple: A boolean indicating if consolidation is possible, and the potentially modified DataFrame.
        If consolidation is not possible and user chooses not to resolve, the original DataFrame is returned.
        If False, prompts the user to resolve violations by keeping the row with the most non-missing values.
        Prints the (group_cols) combinations where consolidation is not possible.
    
    firm_name    |  year    | revenue | ev | employees |
    ----------------------------------------------------
    A            | 2020     | 1e6     | 1e4| NA        |
    A            | 2020     | NA      | NA | 100       |
    
    Is acceptable because it can consolidate to:
    
    firm_name    |  year    | revenue | ev | employees |
    ----------------------------------------------------
    A            | 2020     | 1e6     | 1e4| 100       |
    
    However, we couldn't multiple/contradicting firm_name-year observations, e.g.:
    firm_name    |  year    | revenue | ev | employees |
    ----------------------------------------------------
    A            | 2020     | 1e6     | 1e4| 80        |
    A            | 2020     | 1e7     | NA | 100       |
    
    """
    # Check whether each column has exactly one or zero non-missing observations per group
    def is_valid_group(group):
        return group.nunique(dropna=True) <= 1
    
    grouped = df.groupby(group_cols)[historical_cols].apply(lambda x: x.apply(is_valid_group))
    violations = grouped[~grouped].stack().index.tolist()
    
    if len(violations) == 0:
        return True, df
    else:
        violation_groups = set([(violation[0], violation[1]) for violation in violations])
        print("Cannot reconcile and consolidate the following observations: ")
        for violation_group in violation_groups:
            print(violation_group)
            user_input = input("Resolve violation by keeping the row with most non-missings? (Y/N): ").strip().upper()
            if user_input == 'Y':
                # Create a condition to filter the group with violations dynamically
                condition = (df[group_cols[0]] == violation_group[0])
                for col, val in zip(group_cols[1:], violation_group[1:]):
                    condition &= (df[col] == val)
                
                group = df[condition]
                row_with_most_non_missing = group.loc[group.notnull().sum(axis=1).idxmax()]
                
                # Remove all rows for the violating group from the original DataFrame
                df = df[~condition]
                
                # Add the row with the most non-missing values back to the DataFrame
                df = pd.concat([df, row_with_most_non_missing.to_frame().T], ignore_index=True)
            else:
                return False, df
        return True, df
    

def consolidate_group(group):
    """
    Consolidates a group by forward-filling, backward-filling, and returning the last non-missing value.
        
    Args:
        group (pd.DataFrame): The grouped DataFrame.

    Returns:
        pd.Series: The consolidated series with no missing values.
    """
    return group.ffill().bfill().iloc[-1]


def consolidate_observations(df, group_cols, historical_cols):
    """
    Consolidates the observations in the DataFrame if consolidation is possible.

    This function checks if consolidation is possible by ensuring that there are no conflicting values within
    each group. If consolidation is possible, it applies the consolidate_group function to consolidate the data.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.
        group_cols (list): List of columns to group by (e.g., ['firm_name', 'year']).
        historical_cols (list): List of historical columns to check for consolidation.

    Returns:
        pd.DataFrame: The consolidated DataFrame if consolidation is possible.
        If consolidation is not possible, returns None and prints the conflicting groups.
    """
    consolidation, df = consolidation_is_possible(df=df, group_cols=group_cols, historical_cols=historical_cols)
    if consolidation:
        df_consolidated = df.groupby(group_cols).apply(consolidate_group).reset_index(drop=True)
        print("Consolidation successful")
        return df_consolidated
    else:
        print("Consolidation not possible")
        return df

    
def drop_rows_with_all_missings(df, fields_to_check, verbose=False):
    """
    Drops rows in a DataFrame where all specified fields have missing values.

    This function removes rows from the DataFrame where all the values in the specified fields
    are missing (i.e., `NaN`). Optionally, it prints the number of observations dropped.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
        fields_to_check (list): List of column names to check for missing values.
        verbose (bool, optional): If True, prints the number of observations dropped. Default is False.

    Returns:
        pd.DataFrame: The cleaned DataFrame with rows dropped where all specified fields were missing.
    """
    n1 = len(df)
    df = df.dropna(subset=fields_to_check, how='all')
    n2 = len(df)
    
    if verbose:
        print(f"{n1 - n2} observations dropped because all historical data was missing.")
    
    df = df.reset_index(drop=True)
    return df


def clean_raw_data_from_load(df, group_cols, historical_cols):
    df = standardise_missing_values(df)
    df = mask_non_reported_co2e(df)
    df = consolidate_observations(df, group_cols=group_cols, historical_cols=historical_cols)
    df = drop_rows_with_all_missings(df, fields_to_check=historical_cols)
    return df


def get_mcap(source):
    mcap = pd.read_excel(source, sheet_name='market_cap')
    mcap.columns = ['instrument', 'year', 'mcap']
    mcap['year'] = mcap['year'].dt.year.astype(int)
    return mcap

if __name__=="__main__":
    # Load the data
    file_path = 'data/ftse_global_allcap.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
   
    # Basic cleaning
    grp_cols = ['instrument', 'financial_year']
    revenue_index = data.columns.get_loc("revenue")
    historic_cols = list(data.columns[revenue_index:])
    data = standardise_missing_values(data)
    data = mask_non_reported_co2e(data)
    data = clean_raw_data_from_load(data, group_cols=grp_cols, historical_cols=historic_cols)
    data['year'] = data['financial_year'].str.replace('FY', '').astype(int)
    data = data[(data['year'] >= 2016) & (data['year'] <= 2023)]

    # # mcap = get_mcap('data/ftse_350/databook.xlsx')
    # # data = data.merge(mcap, on=['instrument', 'year'], how='left')
    
    # Save
    data.to_csv('data/ftse_global_allcap_clean.csv', index=False)
