import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

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
    # Identify columns related to individual Scope 3 categories
    s3_category_cols = df.loc[:, 's3_purchased_goods_cat1':'s3_investments_cat15'].columns
    
    # Calculate the sum across Scope 3 categories
    df['sum_across_s3_categories'] = df[s3_category_cols].sum(axis=1)
    
    # Mark the total sum as missing if it is zero
    df.loc[df['sum_across_s3_categories'] == 0, 'sum_across_s3_categories'] = np.nan
    
    # Determine where both are missing, one is missing, or neither is missing
    both_missing = np.logical_and(np.isnan(df['sum_across_s3_categories']), np.isnan(df['s3_co2e']))
    one_missing = np.isnan(df['sum_across_s3_categories']) != np.isnan(df['s3_co2e'])
    total_present_cat_missing = np.logical_and(np.isnan(df['sum_across_s3_categories']), ~np.isnan(df['s3_co2e']))
    cat_present_total_missing = np.logical_and(~np.isnan(df['sum_across_s3_categories']), np.isnan(df['s3_co2e']))
    neither_missing = np.logical_and(~np.isnan(df['sum_across_s3_categories']), ~np.isnan(df['s3_co2e']))
    
    # Compare the sums and the reported total within the specified tolerance
    df['discrepancy_perc'] = 100 * abs(df['sum_across_s3_categories'] - df['s3_co2e']) / df['s3_co2e']
    df['is_sum_equal_to_s3_co2e'] = df['discrepancy_perc'] <= tolerance

    # Identify non-matching totals beyond the specified tolerance
    non_matching_totals = df.loc[np.logical_and(df['is_sum_equal_to_s3_co2e'] == False, neither_missing), 
                                 observation_group + ['sum_across_s3_categories', 's3_co2e', 'discrepancy_perc']]
    
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

       
def get_firms_per_year(df, verbose=True):
    """
    Computes the number of unique firms per year.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least 'year' and 'firm_name' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns 'year' and 'n_firms' indicating the number of unique firms per year.
    """
    result = pd.DataFrame(data={'n_firms': df.groupby('year')['firm_name'].nunique()}).reset_index()
    if verbose:
        print(result)
    return result


def non_missing_count(df, cols_to_check=['revenue', 'ev', 's1_co2e', 's2_co2e', 's3_co2e'], verbose=True):
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
    s3_cat_cols = df.columns[df.columns.str.contains('^s3_') & (df.columns != 's3_co2e')]
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
    # Compute the number of unique firms per year
    n_firms = get_firms_per_year(df)
    
    # Compute non-missing counts for specified columns
    non_missings = non_missing_count(df)
    
    # Compute the count of firms with at least one non-missing Scope 3 category value per year
    s3_coverage = at_least_one_s3_cat_coverage(df)
    
    # Merge the results
    not_null_prop = n_firms.merge(non_missings, on='year').merge(s3_coverage, on='year')
    
    # Columns to calculate proportions
    columns_to_divide = ['revenue', 'ev', 's1_co2e', 's2_co2e', 's3_co2e', 's3_cat_total']
    
    # Calculate proportions
    not_null_prop[columns_to_divide] = not_null_prop[columns_to_divide].div(not_null_prop['n_firms'], axis=0)

    # Print the summarized data if verbose
    if verbose:
        print("\n===================\nNON-NULL PROPORTIONS PER FIELD\n===================\n")
        print(not_null_prop)
        
    return not_null_prop


def get_new_firms_per_year(df, verbose=True):
    """
    Identifies the first year of non-missing s3_co2e for each firm and counts the number of new firms reporting each year.

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns 'firm_name', 'year', and 's3_co2e'.

    Returns:
        pd.DataFrame: A DataFrame with years as the index and the count of new firms reporting s3_co2e in each year.
    """
    if 'firm_name' not in df.columns or 'year' not in df.columns or 's3_co2e' not in df.columns:
        raise ValueError("DataFrame must contain 'firm_name', 'year', and 's3_co2e' columns")
    
    non_missing_df = df.dropna(subset=['s3_co2e'])
    first_non_missing_year_df = non_missing_df.groupby('firm_name')['year'].min().reset_index()
    first_non_missing_year_df.columns = ['firm_name', 'first_year']
    
    new_firms_per_year = first_non_missing_year_df['first_year'].value_counts().sort_index()
    new_firms_per_year_df = new_firms_per_year.reset_index()
    new_firms_per_year_df.columns = ['year', 'newly_reporting_firms']
    if verbose:
        print(new_firms_per_year_df)
    return new_firms_per_year_df


def get_avg_s3_share_by_industry(df, verbose=True):
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
    s3_reporters = df.loc[all_emissions_reported, ['firm_name', 'year', 'icb_industry_code', 'icb_industry_name', 'icb_supersector_code', 'icb_supersector_name'] + co2e_cols].reset_index(drop=True)
    s3_reporters['carbon_footprint'] = s3_reporters[co2e_cols].sum(axis=1)
    for col in co2e_cols:
        s3_reporters[f'{col}_proportion'] = s3_reporters[col] / s3_reporters['carbon_footprint']
    df = {x:s3_reporters.loc[s3_reporters['icb_industry_name'] == x, 's3_co2e_proportion'].mean() for x in s3_reporters['icb_industry_name'].unique()}
    result = pd.DataFrame.from_dict(df, orient='index', columns=['s3_prop']).reset_index(names='sector').sort_values(by='s3_prop').reset_index(drop=True)
    if verbose:
        print(result)
    return result
    
# See source 18 in notes

def get_s3_prop_by_industry(data, verbose=True):
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
    s3_category_cols = data.loc[:, 's3_purchased_goods_cat1':'s3_investments_cat15'].columns
    s3_upstream_cols = data.loc[:, 's3_purchased_goods_cat1':'s3_leased_assets_cat8'].columns
    s3_downstream_cols = data.loc[:, 's3_distribution_cat9': 's3_investments_cat15'].columns
    
    data = data.copy()
    
    data['sum_across_s3_categories'] = data[s3_category_cols].sum(axis=1)
    data['upstream_sum'] = data[s3_upstream_cols].sum(axis=1)
    data['downstream_sum'] = data[s3_downstream_cols].sum(axis=1)
    
    data.loc[data['sum_across_s3_categories'] == 0, 'sum_across_s3_categories'] = np.nan
    data['discrepancy_perc'] = 100 * abs(data['sum_across_s3_categories'] - data['s3_co2e']) / data['s3_co2e']
    data['is_sum_equal_to_s3_co2e'] = data['discrepancy_perc'] <= 0.5
    data = data[data['is_sum_equal_to_s3_co2e'] == True]
    
    data['carbon_footprint'] = data[['s1_co2e', 's2_co2e', 'upstream_sum', 'downstream_sum']].sum(axis=1)
    data['upstream'] = data['upstream_sum'] / data['carbon_footprint']
    data['downstream'] = data['downstream_sum'] / data['carbon_footprint']
    
    # Group by industry and calculate the mean upstream and downstream shares
    result = data.groupby(['icb_industry_name'])[['upstream', 'downstream']].mean().reset_index()
    
    if verbose:
        print(result)
    return result

def get_coverage_by_mcap():
    # Reported
    # Estimated
    # Missing
    pass

def get_avg_total_ghg_per_year_by_cat():
    pass

    
if __name__=="__main__":
    # Load the data
    data = pd.read_csv('data/ftse_350/clean_from_excel.csv')
    i = 'firm_name'
    t = 'year'
    
    