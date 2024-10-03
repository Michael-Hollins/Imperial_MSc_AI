import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set_style("whitegrid")
from clean_raw_data import get_s3_cat_cols
from clean_raw_data import get_sector_cols
from ml_pipeline import *


def audit_value_types(dt, axis_val=0, verbose=True):
    """
    Audits the types of values (zeros, nulls, and non-null/non-zero values) in a DataFrame along the specified axis.
    
    This function calculates the proportion of zeros, nulls, and non-null/non-zero values in the DataFrame
    along the specified axis. It can either audit by columns (axis_val=0) or by rows (axis_val=1) and
    optionally print a summary.

    Args:
        dt (pd.DataFrame): The input DataFrame to audit.
        axis_val (int, optional): The axis along which to audit, 0 for columns and 1 for rows. Defaults to 0.
        verbose (bool, optional): If True, prints the summary. Default is True.

    Returns:
        pd.DdataFrame: A DataFrame with the proprtion of zeros, nulls, and non-zero/non-null values.
        
    Raises:
        ValueError: If axis_val is not 0 or 1.
        
    Note: 
        - When axis_val=0, the audit is performed column-wise.
        - When axis_val=1, the audit is performed row-wise and grouped by 'year' and 'instrument'.
    """
    if axis_val not in [0, 1]:
        raise ValueError("axis_val must be either 0 (cols) or 1 (rows)")
    
    if axis_val == 0:
        divisor = len(dt)
        index_val = dt.columns
        zeros = dt.isin([0]).sum(axis=axis_val) / divisor
        nulls = dt.isnull().sum(axis=axis_val) / divisor
        all_else = np.logical_and(dt.notnull(), dt != 0).sum(axis=axis_val) / divisor
        summary = pd.DataFrame(data={'zeros': zeros, 'nulls': nulls, 'all_else': all_else}, index=index_val)
    
    elif axis_val == 1:
        group = dt.groupby(['year', 'instrument']) if 'year' in dt.columns and 'instrument' in dt.columns else dt.groupby(level=0)
        def calc_proportions(x):
            total_elements = len(x.columns) * len(x)
            zeros_count = (x == 0).sum().sum()
            nulls_count = x.isnull().sum().sum()
            all_else_count = np.logical_and(x.notnull(), x != 0).sum().sum()
            return pd.Series({'zeros': zeros_count / total_elements, 
                              'nulls': nulls_count / total_elements, 
                              'all_else': all_else_count / total_elements})
        summary = group.apply(calc_proportions)
       
    if verbose:
        print("\n===================\nVALUE AUDIT\n===================\n")
        print(summary)
        
    return summary


def compare_s3_categories_with_total(df, observation_group, tolerance=0.5, verbose=True):
    """
    Compares the total Scope 3 CO2 emissions (s3_co2e) with the sum of individual Scope 3 categories within a specified tolerance.
    
    This function performs the following steps:
    1. Identifies columns related to individual Scope 3 categories.
    2. Calculates the sum of these categories for each row.
    3. Marks the total sum as missing if it is zero.
    4. Compares the calculated sum with the reported total Scope 3 CO2 emissions within a specified tolerance.
    5. Identifies and prints discrepancies, if any, along with summary statistics.

    Args:
        df (pandas.DataFrame): The input DataFrame containing Scope 3 CO2 emission data.
        tolerance (float, optional): The percentage tolerance within which the sums are considered equal. Default is 0.5%.
        verbose (bool, optional): If True, prints detailed comparison results. Default is True.

    Returns:
        pd.DataFrame: A DataFrame containing non-matching totals and discrepancy percentages.

    Notes:
        - The function assumes that individual Scope 3 category columns are contiguous in the DataFrame.
        - It prints detailed statistics about the matching process if verbose is set to True.
    """
    
    # Mark the total sum as missing if it is zero
    df.loc[df['s3_cat_total'] == 0, 's3_cat_total'] = np.nan
    
    # Determine where both are missing, one is missing, or neither is missing
    both_missing = np.logical_and(np.isnan(df['s3_cat_total']), np.isnan(df['s3_co2e']))
    one_missing = np.isnan(df['s3_cat_total']) != np.isnan(df['s3_co2e'])
    total_present_cat_missing = np.logical_and(np.isnan(df['s3_cat_total']), ~np.isnan(df['s3_co2e']))
    cat_present_total_missing = np.logical_and(~np.isnan(df['s3_cat_total']), np.isnan(df['s3_co2e']))
    neither_missing = np.logical_and(~np.isnan(df['s3_cat_total']), ~np.isnan(df['s3_co2e']))
    
    # Compare the sums and the reported total within the specified tolerance
    df['discrepancy_perc'] = 100 * abs(df['s3_cat_total'] - df['s3_co2e']) / df['s3_co2e']
    df['is_sum_equal_to_s3_co2e'] = df['discrepancy_perc'] <= tolerance

    # Identify non-matching totals beyond the specified tolerance
    non_matching_totals = df.loc[np.logical_and(df['is_sum_equal_to_s3_co2e'] == False, neither_missing), 
                                 observation_group + ['s3_cat_total', 's3_co2e', 'discrepancy_perc']]
    
    # Sort by discrepancy percentage
    non_matching_totals.sort_values(by='discrepancy_perc', ascending=False, inplace=True, ignore_index=True)
    
    if verbose:
        print("\n===================\nCOMPARING TOTAL S3 FIELD TO SUM OF S3 CATEGORICAL FIELDS\n===================\n")
        print(f"We begin with {len(df)} observations.\n")
        print(f"{sum(both_missing)} observations ({100 * sum(both_missing) / len(df):.2f}%) have both the total S3 amount and the implied total over the categories missing.\n")
        print(f"{sum(one_missing)} observations ({100 * sum(one_missing) / len(df):.2f}%) have either the total S3 amount missing or the totalled up amount missing.\nThis breaks down into {sum(total_present_cat_missing)} observations that have the total field populated but no category data.\n{sum(cat_present_total_missing)} observations have it the other way around.\n")
        print(f"Therefore we check the correspondence between the reported total and category total for {sum(neither_missing)} ({100 * sum(neither_missing) / len(df):.2f}%) cases.\n")
        print(f"Of these, {sum(df['is_sum_equal_to_s3_co2e'])} observations have a matching total field and implied category total within the tolerance of {tolerance}%.\n")
        print(f"The mean discrepancy for non-matching totals is {np.mean(non_matching_totals['discrepancy_perc']):.2f}% and the median is {np.median(non_matching_totals['discrepancy_perc']):.2f}%\n")
        print("The top ten largest discrepancies are...\n")
        print(non_matching_totals.head(10))
    
    return non_matching_totals

       
def get_instruments_per_year(df, verbose=True):
    """
    Computes the number of unique instruments (some firm names are shared with different instruments) per year.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least 'year' and 'instrument' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns 'year' and 'n_instruments' indicating the number of unique instruments per year.
    """
    result = pd.DataFrame(data={'n_instruments': df.groupby('year')['instrument'].nunique()}).reset_index()
    if verbose:
        print(result)
    return result


def non_missing_count(df, cols_to_check=['revenue', 'mcap', 's1_co2e', 's2_co2e', 's3_co2e'], verbose=True):
    """
    Computes the count of non-missing values for specified columns per year.

    Args:
        df (pd.DataFrame): The input DataFrame containing the columns to check.
        cols_to_check (list, optional): List of column names to check for non-missing values. Default is specified.
    Returns:
        pd.DataFrame: A DataFrame with the count of non-missing values for each specified column per year.
    """
    result = df.groupby('year')[cols_to_check].apply(lambda x: x.notna().sum()).reset_index()
    if verbose:
        print(result)
    return result


def at_least_one_s3_cat_coverage(df, verbose=True):
    """
    Computes the count of firms per year that have at least one non-missing value in the Scope 3 category columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing Scope 3 category columns.

    Returns:
        pd.DataFrame: A DataFrame with columns 'year' and 's3_cat_data' indicating the count of firms with at least
        one non-missing Scope 3 category value per year.
    """
    s3_cat_cols = get_s3_cat_cols(df)
    df['s3_cat_total'] = df[s3_cat_cols].notna().any(axis=1).astype(int)
    non_null_counts_per_year = df.groupby('year')['s3_cat_total'].sum().reset_index()
    if verbose:
        print(non_null_counts_per_year)
    return non_null_counts_per_year


def summarise_data_coverage(df, verbose=True):
    """
    Summarizes data coverage by computing the proportion of non-missing values for specified columns per year.

    This function performs the following steps:
    1. Computes the number of unique firms per year.
    2. Computes the count of non-missing values for specified columns per year.
    3. Computes the count of firms per year with at least one non-missing Scope 3 category value.
    4. Merges the computed metrics and calculates the proportion of non-missing values.
    5. Prints the summarized data coverage if verbose is True.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the relevant columns.
        verbose (bool, optional): If True, prints the summarized data coverage. Default is True.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the data coverage with proportions of non-missing values.
    """
    # Compute the number of unique instruments per year
    n_instruments = get_instruments_per_year(df)
    
    # Compute non-missing counts for specified columns
    non_missings = non_missing_count(df)
    
    # Compute the count of instruments with at least one non-missing Scope 3 category value per year
    s3_coverage = at_least_one_s3_cat_coverage(df)
    
    # Merge the results
    not_null_prop = n_instruments.merge(non_missings, on='year').merge(s3_coverage, on='year')
    
    # Columns to calculate proportions
    columns_to_divide = ['revenue',  's1_co2e', 's2_co2e', 's3_co2e', 's3_cat_total']
    
    # Calculate proportions
    not_null_prop[columns_to_divide] = not_null_prop[columns_to_divide].div(not_null_prop['n_instruments'], axis=0)

    # Print the summarized data if verbose
    if verbose:
        print("\n===================\nNON-NULL PROPORTIONS PER FIELD\n===================\n")
        print(not_null_prop)
        
    return not_null_prop


def get_new_instruments_per_year(df, verbose=True):
    """
    Identifies the first year of non-missing s3_co2e for each instrument and counts the number of new instruments reporting each year.

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns 'instrument', 'year', and 's3_co2e'.

    Returns:
        pd.DataFrame: A DataFrame with years as the index and the count of new instruments reporting s3_co2e in each year.
    """
    if 'instrument' not in df.columns or 'year' not in df.columns or 's3_co2e' not in df.columns:
        raise ValueError("DataFrame must contain 'instrument', 'year', and 's3_co2e' columns")
    
    non_missing_df = df.dropna(subset=['s3_co2e'])
    first_non_missing_year_df = non_missing_df.groupby('instrument')['year'].min().reset_index()
    first_non_missing_year_df.columns = ['instrument', 'first_year']
    
    new_instruments_per_year = first_non_missing_year_df['first_year'].value_counts().sort_index()
    new_instruments_per_year_df = new_instruments_per_year.reset_index()
    new_instruments_per_year_df.columns = ['year', 'newly_reporting_instruments']
    if verbose:
        print(new_instruments_per_year_df)
    return new_instruments_per_year_df


def get_avg_s3_share_by_industry(df, industry_col, verbose=True):
    """
    Calculates the average share of Scope 3 CO2e emissions by industry.

    This function filters the input DataFrame to include only rows where all CO2e columns ('s1_co2e', 's2_co2e', 's3_co2e') are reported.
    It then calculates the proportion of each CO2e type relative to the total carbon footprint for each row.
    Finally, it computes the average proportion of Scope 3 CO2e emissions for each industry and returns the results in a sorted DataFrame.
    Note that this function uses the TOTAL reported S3 emissions and not the category-level data.

    Args:
        df (pd.DataFrame): The input DataFrame containing CO2e emissions data and industry information.
        verbose (bool): If True, prints the resulting DataFrame. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with the average proportion of Scope 3 CO2e emissions for each industry,
                      sorted by the proportion in ascending order. The DataFrame contains columns 'sector' and 's3_prop'.
    """
    co2e_cols = ['s1_co2e', 's2_co2e', 's3_co2e']
    all_emissions_reported = df[co2e_cols].notnull().all(axis=1)
    s3_reporters = df.loc[all_emissions_reported, ['firm_name', 'year'] + get_sector_cols() + co2e_cols].reset_index(drop=True)
    s3_reporters['carbon_footprint'] = s3_reporters[co2e_cols].sum(axis=1)
    for col in co2e_cols:
        s3_reporters[f'{col}_proportion'] = s3_reporters[col] / s3_reporters['carbon_footprint']
    df = {x:s3_reporters.loc[s3_reporters[industry_col] == x, 's3_co2e_proportion'].median() for x in s3_reporters[industry_col].unique()}
    result = pd.DataFrame.from_dict(df, orient='index', columns=['s3_prop']).reset_index(names='sector').sort_values(by='s3_prop').reset_index(drop=True)
    if verbose:
        print(result)
    return result
    

def get_s3_prop_by_industry(df, industry_col, verbose=True):
    """
    Calculates the average upstream and downstream Scope 3 CO2e emissions share by industry.

    This function processes the input DataFrame to calculate the sum of Scope 3 CO2e emissions 
    across various categories. It ensures that the summed categories match the reported Scope 3 
    CO2e emissions within a 0.5% discrepancy. It then calculates the total carbon footprint 
    and the relative shares of upstream and downstream emissions. Finally, it computes the 
    average upstream and downstream shares for each industry.

    Args:
        data (pd.DataFrame): The input DataFrame containing CO2e emissions data and industry information.
        verbose (bool): If True, prints the resulting DataFrame. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with the average upstream and downstream Scope 3 CO2e emissions share 
                      for each industry, containing columns 'icb_industry_name', 'upstream_share', and 'downstream_share'.
    """
    s3_category_cols = df.loc[:, 's3_purchased_goods_cat1':'s3_investments_cat15'].columns
    s3_upstream_cols = df.loc[:, 's3_purchased_goods_cat1':'s3_leased_assets_cat8'].columns
    s3_downstream_cols = df.loc[:, 's3_distribution_cat9': 's3_investments_cat15'].columns
    
    df = df.copy()
    
    df['sum_across_s3_categories'] = df[s3_category_cols].sum(axis=1)
    df['upstream_sum'] = df[s3_upstream_cols].sum(axis=1)
    df['downstream_sum'] = df[s3_downstream_cols].sum(axis=1)
    
    df.loc[df['sum_across_s3_categories'] == 0, 'sum_across_s3_categories'] = np.nan
    df['discrepancy_perc'] = 100 * abs(df['sum_across_s3_categories'] - df['s3_co2e']) / df['s3_co2e']
    df['is_sum_equal_to_s3_co2e'] = df['discrepancy_perc'] <= 0.5
    df = df[df['is_sum_equal_to_s3_co2e'] == True]
    
    df['carbon_footprint'] = df[['s1_co2e', 's2_co2e', 'upstream_sum', 'downstream_sum']].sum(axis=1)
    df['upstream'] = df['upstream_sum'] / df['carbon_footprint']
    df['downstream'] = df['downstream_sum'] / df['carbon_footprint']
    
    # Group by industry and calculate the mean upstream and downstream shares
    result = df.groupby([industry_col])[['upstream', 'downstream']].mean().reset_index()
    
    if verbose:
        print(result)
    return result


def get_avg_num_s3_cats_reported(df, year_lower=2020, year_upper=2022, verbose=True):
    """
    Calculates the median number of Scope 3 (S3) categories reported by firms within a specified year range.

    This function filters the DataFrame for the specified year range and identifies columns corresponding to S3 categories.
    It then calculates the number of non-missing S3 categories reported by each firm, considers only those firms that
    report at least one S3 category, and computes the median number of S3 categories reported.

    Args:
        df (pandas.DataFrame): The input DataFrame containing firm data with S3 categories.
        year_lower (int): The lower bound of the year range (inclusive).
        year_upper (int): The upper bound of the year range (inclusive).
        verbose (bool, optional): If True, prints the median number of S3 categories reported. Default is True.

    Returns:
        float: The median number of S3 categories reported by firms that report at least one category.
    """
    df = df[(df['year'] >= year_lower) & (df['year'] <= year_upper)]
    s3_cat_cols = get_s3_cat_cols(df)
    res = pd.Series(df.loc[:, s3_cat_cols].notna().sum(axis=1).astype(int))  
    res = res[res > 0]
    res = res.median()
    if verbose:
        print(f"The median number of S3 categories reported by firms between {year_lower} and {year_upper} for those firms reporting at least one category is {res}")
    return res
    

def sector_breakdown_for_s3_cat_data(df, verbose=True):
    """
    Generate a sector breakdown for Scope 3 category data.

    This function processes the given DataFrame to filter and count the occurrences of different sectors
    based on Scope 3 category data. It filters the data for the years 2020 to 2022, and for rows where
    at least one Scope 3 category (excluding 's3_co2e') is reported. The function then counts the occurrences
    of unique combinations of industry codes, supersector codes, and sector codes, and sorts the results.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing Scope 3 emissions data along with industry and sector information.
                         Expected columns include 'year', 'icb_industry_code', 'icb_industry_name', 'icb_supersector_code',
                         'icb_supersector_name', 'icb_sector_code', 'icb_sector_name', and various 's3_' columns.
    verbose (bool): If True, prints the sorted DataFrame. Default is True.

    Returns:
    pd.DataFrame: A DataFrame containing the counts of unique combinations of industry codes, supersector codes, and sector codes,
                  sorted by 'icb_industry_code', 'icb_supersector_code', and 'icb_sector_code'.
    """
    sector_cols = get_sector_cols()
    s3_cat_cols = get_s3_cat_cols(df)
    df['s3_cat_total'] = df[s3_cat_cols].notna().any(axis=1).astype(int)
    df = df[df['s1_co2e'].notnull()]
    counts = df[sector_cols].value_counts()
    counts_df = counts.reset_index(name='count')
    sorted_counts_df = counts_df.sort_values(by=['econ_sector_code', 'business_sector_code', 'industry_group_sector_code', 'industry_sector_code', 'activity_sector_code']).reset_index(drop=True)
    if verbose:
        print(sorted_counts_df)
    return sorted_counts_df


def describe_data(df):
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
    dtypes = df.dtypes
    newdf = df.select_dtypes(include=numeric_types)
    total = df.isnull().count()
    missing = df.isnull().sum()
    available = total-missing
    missing_percent = np.round(missing/total, 2)
    unique = df.nunique()
    min_value = np.round(newdf.min(), 0)
    max_value = np.round(newdf.max(), 0)
    mean_value = np.round(newdf.mean(), 0)
    median_value = np.round(newdf.median(), 0)
    quantile_95 = np.round(newdf.quantile(0.95), 0)
    quantile_5 = np.round(newdf.quantile(0.05), 0)
    quantile_25 = np.round(newdf.quantile(0.25), 0)
    quantile_75 = np.round(newdf.quantile(0.75), 0)
    quantile_10 = np.round(newdf.quantile(0.10), 0)
    quantile_90 = np.round(newdf.quantile(0.90), 0)
    std = np.round(newdf.std(), 0)
    kurtosis = np.round(newdf.kurtosis(), 0)
    skew = np.round(newdf.skew(), 0)
    
    Summary = pd.concat([dtypes,total,missing,available,missing_percent,unique, min_value,\
						 max_value,mean_value,median_value,std,kurtosis,skew,\
						 quantile_5,quantile_95,\
						 quantile_25,quantile_75,\
						 quantile_10, quantile_90,\
						 ],axis=1,keys=["Types","Total","Missing","Available",\
						 "Missing_Percent","Unique","Min","Max","Mean","Median","Std Error"\
						 ,"Kurtosis","Skewness","Quantile5","Quantile95",\
						 "Quantile25","Quantile75","Quantile10","Quantile90"])
    Summary.reset_index(level=0, inplace=True)
    return (Summary)
