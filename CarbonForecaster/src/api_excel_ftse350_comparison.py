from load_raw_data import load_from_excel
from clean_raw_data import clean_raw_data_from_load
import pandas as pd
from load_raw_data import col_mapping

"""
The purpose of this script is to understand the best way to load the data. 
We use the FTSE350 as our sample, and pull from the API and via Excel. 
Where the outputs disagree, we print the values which we can compare against the WorkSpace terminal.
Our findings show that the API is the superior choice as it more closely aligns with the terminal.

The API query uses the load_raw_data.py file and the following call:
save_data_from_api(universe=get_ftse_350(),
                  interval='quarterly',
                  start_date='2014-01-01',
                  end_date='2025-01-01',
                  chunk_size=50)
                  
The Excel file is created using:
static data: =@RDP.Data(universe!$A$2:$A$351,static_fields!$A$2:$A$8,"CH=Fd RH=IN",$A$1)
historical data: =@RDP.Data(universe!$A$2:$A$351,historical_fields!$A$2:$A$38,"Period=FY0 Frq=FY SDate=0 EDate=-14 CH=Fd RH=date;IN NULL=NA",$A$1)

This script can also be thought of as a high-level test for the key functions in the load_raw_data.py file for the API call
as it checks the correspondence with the similar/same call using Excel.
"""

def identify_common_keys(df1, df2, key_cols):
    """
    Identifies common keys between two dataframes and returns the keys present in both.

    Parameters:
    df1 (pd.DataFrame): First dataframe.
    df2 (pd.DataFrame): Second dataframe.
    key_cols (list): List of columns to use as keys for merging.

    Returns:
    pd.DataFrame: DataFrame with keys common to both df1 and df2.
    """
    merged_keys = df1[key_cols].merge(df2[key_cols], on=key_cols, how='inner')
    return merged_keys.drop_duplicates()


def filter_dataframes(df1, df2, common_keys, key_cols):
    """
    Filters two dataframes to only include rows with keys present in the common keys dataframe.

    Parameters:
    df1 (pd.DataFrame): First dataframe.
    df2 (pd.DataFrame): Second dataframe.
    common_keys (pd.DataFrame): DataFrame with keys common to both df1 and df2.
    key_cols (list): List of columns to use as keys for filtering.

    Returns:
    pd.DataFrame: Filtered first dataframe.
    pd.DataFrame: Filtered second dataframe.
    
    Additionally:
    Prints the number of rows dropped from each dataframe.
    """
    n_df1 = len(df1)
    n_df2 = len(df2)
    df1_filtered = df1.merge(common_keys, on=key_cols, how='inner')
    df2_filtered = df2.merge(common_keys, on=key_cols, how='inner')
    n_df1_filtered = len(df1_filtered)
    n_df2_filtered = len(df2_filtered)
    if n_df1 > n_df1_filtered:
        print(f"Dropped {n_df1 - n_df1_filtered} row(s) from df1")
    if n_df2 > n_df2_filtered:
        print(f"Dropped {n_df2 - n_df2_filtered} row(s) from df2")
    return df1_filtered, df2_filtered


def compare_vals(a, b):
    """
    Compares two pandas Series element-wise, returning a boolean Series
    indicating where the values are equal or both are NaN.
    
    Args:
        a (pd.Series): The first Series to compare.
        b (pd.Series): The second Series to compare.
    
    Returns:
        pd.Series: A boolean Series indicating where the values are equal or both are NaN.
    """
    return (a == b) | (pd.isnull(a) & pd.isnull(b))


def compare_dataframes(df1, df2, cols_to_check):
    """
    Compare two dataframes row-wise and return the indices of rows that do not match.

    Parameters:
    df1 (pd.DataFrame): First dataframe.
    df2 (pd.DataFrame): Second dataframe.
    cols_to_check (list): List of columns to check for matching.

    Returns:
    list: Indices of rows that do not match.
    """
    mismatched_indices = []
    
    for i in range(len(df1)):
        if not compare_vals(df1[cols_to_check].iloc[i], df2[cols_to_check].iloc[i]).all():
            mismatched_indices.append(i)

    return mismatched_indices


def inspect_non_matching_rows(df1, df2, indices, firm_details):
    """
    Inspects and prints non-matching rows between two DataFrames for specified indices.

    This function compares rows from two DataFrames at specified indices and prints the columns
    where the values do not match. It is useful for inspecting and understanding discrepancies
    between two DataFrames.

    Args:
        df1 (pandas.DataFrame): The first DataFrame to compare.
        df2 (pandas.DataFrame): The second DataFrame to compare.
        indices (list): A list of row indices to inspect for non-matching values.
        firm_details (list): A list of column names to always include in the output for context.
    """
    for i in indices:
        
        series = compare_vals(df1.iloc[i], df2.iloc[i])
        non_matching_cols = series[series == False].index.tolist()
        
        print(df1[firm_details + non_matching_cols].iloc[i])
        print('===========================')
        print(df2[firm_details + non_matching_cols].iloc[i])
        input("Press Enter to continue to the next row...")
        

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
                      for matching 'instrument' and 'revenue' values from the two data sources.
    """
    df_excel = load_from_excel(excel_file_path, 'static_data', 'historical_data')
    df_excel = df_excel[['instrument', 'absolute_financial_year', 'revenue', 'source_filing_date']].dropna()
    
    df_api = pd.read_pickle(api_file_path)
    df_api = df_api[['instrument', 'relative_financial_year', 'revenue', 'source_filing_date']].dropna()
    
    firms_by_year = df_excel.merge(df_api, on=['instrument', 'revenue', 'source_filing_date'])
    return firms_by_year[['instrument', 'source_filing_date', 'absolute_financial_year', 'relative_financial_year']]


if __name__=="__main__":
    excel_file_path='data/ftse_350/databook.xlsx'
    api_file_path='data/ftse_350/api_call.pkl'
    historic_cols = list(col_mapping.values())[9:]
    df_financial_years = match_absolute_and_relative_financial_years(excel_file_path, api_file_path)
    
    df_api = pd.read_pickle(api_file_path)
    df_api = clean_raw_data_from_load(df_api, group_cols=['instrument', 'relative_financial_year'], historical_cols=historic_cols)
    df_api = df_api.merge(df_financial_years, how='inner', on=['instrument', 'relative_financial_year', 'source_filing_date'])
    df_api.sort_values(by=['instrument', 'source_filing_date'], inplace=True)

    df_excel = load_from_excel(excel_file_path, static_sheet_name='static_data', timeseries_sheet_name='historical_data')
    df_excel = clean_raw_data_from_load(df_excel, group_cols=['instrument', 'absolute_financial_year'], historical_cols=historic_cols)
    df_excel = df_excel.merge(df_financial_years, how='inner', on=['instrument', 'absolute_financial_year', 'source_filing_date'])
    df_excel.sort_values(by=['instrument', 'source_filing_date'], inplace=True)
    df_excel = df_excel[list(df_api.columns)]
    
    mismatched_indices = compare_dataframes(df_api, df_excel, cols_to_check=historic_cols)
    inspect_non_matching_rows(df_api, df_excel, mismatched_indices, firm_details=['firm_name', 'source_filing_date'])
    
    # historical_cols = list(col_mapping.values())[9:]
    # observations = ['firm_name', 'financial_year']
#     df_api = clean_raw_data_from_load(df_api, group_cols=observations, historical_cols=historical_cols)
#     df_api = df_api[(df_api['financial_year'] >= 2015) & (df_api['financial_year'] <= 2023)]
    
#     common_firm_year_combinations = identify_common_keys(df_api, df_excel, key_cols=observations)
#     df_api, df_excel = filter_dataframes(df_api, df_excel, common_keys=common_firm_year_combinations, key_cols=observations)
    
#     mismatched_indices = compare_dataframes(df_api, df_excel, cols_to_check=historical_cols)
    
# #     # Inspect each and compare against the WorkSpace Terminal values
#     inspect_non_matching_rows(df_api, df_excel, mismatched_indices, firm_details=observations)
    
#     # id    | Firm              | Mismatch        | Correct df
#     # ------------------------------------------------------

    
    
    

    
    
     