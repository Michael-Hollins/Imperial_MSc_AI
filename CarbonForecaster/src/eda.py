import pandas as pd
import numpy as np

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


def compare_s3_categories_with_total(df, tolerance=0.5, verbose=True):
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
                                 ['date_val', 'instrument', 'firm_name', 'sum_across_s3_categories', 's3_co2e', 'discrepancy_perc']]
    
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


def subset_and_fill_missings(df, year_lower, year_upper):
    """
    Processes the input DataFrame by sorting, filling missing values, and subsetting based on the given year range.

    This function performs the following steps:
    1. Sorts the DataFrame by 'firm_name' and 'year'.
    2. For each group of 'firm_name' and 'year', it forward-fills and then backward-fills missing values, 
       ensuring that the last non-missing value is kept.
    3. Subsets the DataFrame to include only rows where 'year' is within the specified range [year_lower, year_upper].

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns 'firm_name' and 'year'.
        year_lower (int): The lower bound of the year range for subsetting.
        year_upper (int): The upper bound of the year range for subsetting.

    Returns:
        pandas.DataFrame: The processed DataFrame with missing values filled and subset based on the specified year range.
    """
    df = df.sort_values(by=['firm_name', 'year']).reset_index(drop=True)
    
    def get_last_non_missing(group):
        return group.ffill().bfill().iloc[-1]
    
    df = df.groupby([df['firm_name'],  df['year']]).apply(get_last_non_missing).reset_index(drop=True)
    df = df.loc[(df['year'] >= year_lower) & (df['year'] <= year_upper)]
    return df

        
def get_firms_per_year(df):
    """
    Computes the number of unique firms per year.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least 'year' and 'firm_name' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns 'year' and 'n_firms' indicating the number of unique firms per year.
    """
    return pd.DataFrame(data={'n_firms': df.groupby('year')['firm_name'].nunique()}).reset_index()


def non_missing_count(df, cols_to_check=['revenue', 'ev', 's1_co2e', 's2_co2e', 's3_co2e']):
    """
    Computes the count of non-missing values for specified columns per year.

    Args:
        df (pd.DataFrame): The input DataFrame containing the columns to check.
        cols_to_check (list, optional): List of column names to check for non-missing values. Default is specified.
    Returns:
        pd.DataFrame: A DataFrame with the count of non-missing values for each specified column per year.
    """
    return df.groupby('year')[cols_to_check].apply(lambda x: x.notna().sum()).reset_index()


def at_least_one_s3_cat_coverage(df):
    """
    Computes the count of firms per year that have at least one non-missing value in the Scope 3 category columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing Scope 3 category columns.

    Returns:
        pd.DataFrame: A DataFrame with columns 'year' and 's3_cat_data' indicating the count of firms with at least
        one non-missing Scope 3 category value per year.
    """
    s3_cat_cols = df.columns[df.columns.str.contains('^s3_') & (df.columns != 's3_co2e')]
    df['s3_cat_data'] = df[s3_cat_cols].notna().any(axis=1).astype(int)
    non_null_counts_per_year = df.groupby('year')['s3_cat_data'].sum().reset_index()
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
    columns_to_divide = ['revenue', 'ev', 's1_co2e', 's2_co2e', 's3_co2e', 's3_cat_data']
    
    # Calculate proportions
    not_null_prop[columns_to_divide] = not_null_prop[columns_to_divide].div(not_null_prop['n_firms'], axis=0)

    # Print the summarized data if verbose
    if verbose:
        print("\n===================\nNON-NULL PROPORTIONS PER FIELD\n===================\n")
        print(not_null_prop)
        
    return not_null_prop

def get_new_firms_per_year():
    pass


# See source 18 in notes

def get_coverage_by_mcap():
    # Reported
    # Estimated
    # Missing
    pass

def get_avg_total_ghg_per_year_by_cat():
    pass

def run_headline_diagnositcs():
    summarise_data_coverage(data)
    audit_value_types(data, axis_val=0)
    compare_s3_categories_with_total(data)
    
if __name__=="__main__":
    data = pd.read_pickle('data/mock_universe/mock_universe.pkl')
    data = subset_and_fill_missings(data, year_lower=2014, year_upper=2022)
    #print(data['co2e_method'].value_counts())
    run_headline_diagnositcs()
    #data.to_csv('data/sample.csv')