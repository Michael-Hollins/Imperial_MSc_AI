from load_raw_data import load_from_excel
from clean_raw_data import process_missings_from_load
from clean_raw_data import keep_last_observation_per_year
import pandas as pd


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
        print(f"Dropped {n_df1 - n_df1_filtered} row(s) from {df1}")
    if n_df2 > n_df2_filtered:
        print(f"Dropped {n_df2 - n_df2_filtered} row(s) from {df2}")
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
        

if __name__=="__main__":
    # Read in the data from the API call in load_raw_data
    df_api = pd.read_pickle('data/ftse_350/ftse350_from_api.pkl')
    df_api = process_missings_from_load(df_api, year_lower=2015, year_upper=2024, verbose=False)
    df_api = keep_last_observation_per_year(df_api)
    
    # Read in the data from the Excel WorkSpace plugin
    df_excel = load_from_excel('data/ftse_350/databook.xlsx', 'static_data', 'historical_data')
    df_excel = process_missings_from_load(df_excel, year_lower=2015, year_upper=2024, verbose=False)
    df_excel = keep_last_observation_per_year(df_excel)
    
    # Keep observations that are common between them so we can compare values
    common_firm_year_combinations = identify_common_keys(df_api, df_excel, key_cols=['firm_name', 'year'])
    df_api, df_excel = filter_dataframes(df_api, df_excel, common_keys=common_firm_year_combinations, key_cols=['firm_name', 'year'])
    
    firm_details = ['firm_name', 'year']
    financials = ['revenue', 'ev', 'employees', 'mcap', 'ebit', 'net_cash_flow', 'net_ppe', 'cogs', 'intangible_assets', 'lt_debt']
    production = ['prod_crude_oil', 'prod_natural_gas_liquids', 'prod_nat_gas', 'waste']
    co2e = ['s1_co2e', 's2_co2e', 's3_co2e', 'co2e_method']
    cols_to_check = firm_details + financials + production + co2e
    
    mismatched_indices = compare_dataframes(df_api, df_excel, cols_to_check)
    
    # Inspect each and compare against the WorkSpace Terminal values
    inspect_non_matching_rows(df_api, df_excel, mismatched_indices, firm_details)
    
    # id    | Firm              | Mismatch        | Correct df
    # ------------------------------------------------------
    # 21    | 4Imprint          | Financials      | API
    # 27    | 4Imprint          | Financials      | API
    # 232   | Astom Martin      | CO2e            | API
    # 342   | Baillie Gifford JP| CO2e            | API
    # 343   | Baillie Gifford JP| CO2e            | API
    # 411   | Barclays PLC      | Production      | API
    # 593   | Burberry Group    | CO2e            | API
    # 649   | Carnival PLC      | Production      | API
    # 675   | Chemring Group    | COGS            | API
    # 854   | Direct Line       | Financials      | Excel
    # 855   | Direct Line       | Financials      | API
    # 1254  | Greggs PLC        | Financials      | API
    # 1260  | Greggs PLC        | Financials      | API
    # 1401  | Hilton Food Group | Financials      | API
    # 1407  | Hilton Food Group | Financials      | API
    # 1505  | ITV               | CO2e            | API
    # 1591  | International...  | CO2e            | API
    # 1610  | International...  | CO2e            | Unsure 
    # 1806  | Keller Group PLC  | CO2e            | API
    # 1854  | Law Debenture     | CO2e            | API
    # 1957  | Melrose Industries| CO2e            | API
    # 2026  | Monks Investment  | Financials      | Excel (for 2024)
    # 2273  | Persimmon PLC     | CO2e            | API
    # 2463  | Renewables Infra  | CO2e            | API
    # 2464  | Renewables Infra  | CO2e            | API
    # 2492  | Rentokil Initial  | CO2e            | API
    # 2509  | Rio Tinto         | CO2e            | API
    # 2510  | Rio Tinto         | CO2e            | API
    # 2540  | Ruffer Investment | CO2e            | API
    # 2582  | Safestore Holdings| Financials      | API
    # 2829  | Syncona Ltd       | CO2e            | API
    # 3141  | Weir Group        | Financials/CO2e | API
    
    
    

    
    
     