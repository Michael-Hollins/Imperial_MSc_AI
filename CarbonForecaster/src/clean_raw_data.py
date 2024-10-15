import numpy as np
import pandas as pd
import pickle
import re
from scipy.stats.mstats import winsorize

# Define country categories, source: https://www.imf.org/en/Publications/WEO/weo-database/2023/April/groups-and-aggregates
country_classification = {
    'AD': 'Advanced',  # Andorra
    'AU': 'Advanced',  # Australia
    'AT': 'Advanced',  # Austria
    'BE': 'Advanced',  # Belgium
    'CA': 'Advanced',  # Canada
    'HR': 'Advanced',  # Croatia
    'CY': 'Advanced',  # Cyprus
    'CZ': 'Advanced',  # Czech Republic
    'DK': 'Advanced',  # Denmark
    'EE': 'Advanced',  # Estonia
    'FI': 'Advanced',  # Finland
    'FR': 'Advanced',  # France
    'DE': 'Advanced',  # Germany
    'GR': 'Advanced',  # Greece
    'HK': 'Advanced',  # Hong Kong SAR
    'HU': 'Advanced',  # Hungary -- added manually
    'IS': 'Advanced',  # Iceland
    'IE': 'Advanced',  # Ireland
    'IL': 'Advanced',  # Israel
    'IT': 'Advanced',  # Italy
    'JP': 'Advanced',  # Japan
    'KR': 'Advanced',  # Korea
    'LV': 'Advanced',  # Latvia
    'LT': 'Advanced',  # Lithuania
    'LU': 'Advanced',  # Luxembourg
    'MO': 'Advanced',  # Macao SAR
    'MT': 'Advanced',  # Malta
    'NL': 'Advanced',  # Netherlands
    'NZ': 'Advanced',  # New Zealand
    'NO': 'Advanced',  # Norway
    'PT': 'Advanced',  # Portugal
    'PR': 'Advanced',  # Puerto Rico
    'SM': 'Advanced',  # San Marino
    'SG': 'Advanced',  # Singapore
    'SK': 'Advanced',  # Slovak Republic
    'SI': 'Advanced',  # Slovenia
    'ES': 'Advanced',  # Spain
    'SE': 'Advanced',  # Sweden
    'CH': 'Advanced',  # Switzerland
    'TW': 'Advanced',  # Taiwan Province of China
    'GB': 'Advanced',  # United Kingdom
    'US': 'Advanced',  # United States
    # Add more countries classified as 'Developing' or 'Emerging'
    'AF': 'Developing',  # Afghanistan
    'AL': 'Developing',  # Albania
    'DZ': 'Developing',  # Algeria
    'AO': 'Developing',  # Angola
    'AR': 'Developing',  # Argentina
    'AM': 'Developing',  # Armenia
    'AZ': 'Developing',  # Azerbaijan
    'BD': 'Developing',  # Bangladesh
    'BY': 'Developing',  # Belarus
    'BJ': 'Developing',  # Benin
    'BM': 'Developing',  # Bermuda -- added manually
    'BT': 'Developing',  # Bhutan
    'BO': 'Developing',  # Bolivia
    'BA': 'Developing',  # Bosnia and Herzegovina
    'BW': 'Developing',  # Botswana
    'BR': 'Developing',  # Brazil
    'BG': 'Developing',  # Bulgaria
    'BF': 'Developing',  # Burkina Faso
    'BI': 'Developing',  # Burundi
    'KH': 'Developing',  # Cambodia
    'CM': 'Developing',  # Cameroon
    'CV': 'Developing',  # Cabo Verde
    'KY': 'Developing',  # Cayman Islands -- added manually
    'CF': 'Developing',  # Central African Republic
    'TD': 'Developing',  # Chad
    'CL': 'Developing',  # Chile
    'CN': 'Developing',  # China
    'CO': 'Developing',  # Colombia
    'KM': 'Developing',  # Comoros
    'CG': 'Developing',  # Republic of Congo
    'CR': 'Developing',  # Costa Rica
    'CI': 'Developing',  # Côte d'Ivoire
    'DJ': 'Developing',  # Djibouti
    'DM': 'Developing',  # Dominica
    'DO': 'Developing',  # Dominican Republic
    'EC': 'Developing',  # Ecuador
    'EG': 'Developing',  # Egypt
    'SV': 'Developing',  # El Salvador
    'ER': 'Developing',  # Eritrea
    'ET': 'Developing',  # Ethiopia
    'FJ': 'Developing',  # Fiji
    'GA': 'Developing',  # Gabon
    'GM': 'Developing',  # The Gambia
    'GH': 'Developing',  # Ghana
    'GT': 'Developing',  # Guatemala
    'GN': 'Developing',  # Guinea
    'GY': 'Developing',  # Guyana
    'HT': 'Developing',  # Haiti
    'HN': 'Developing',  # Honduras
    'IN': 'Developing',  # India
    'ID': 'Developing',  # Indonesia
    'IR': 'Developing',  # Iran
    'IQ': 'Developing',  # Iraq
    'JM': 'Developing',  # Jamaica
    'JO': 'Developing',  # Jordan
    'KZ': 'Developing',  # Kazakhstan
    'KE': 'Developing',  # Kenya
    'KW': 'Developing',  # Kuwait
    'KG': 'Developing',  # Kyrgyz Republic
    'LA': 'Developing',  # Lao P.D.R.
    'LB': 'Developing',  # Lebanon
    'LS': 'Developing',  # Lesotho
    'LR': 'Developing',  # Liberia
    'LY': 'Developing',  # Libya
    'MG': 'Developing',  # Madagascar
    'MW': 'Developing',  # Malawi
    'MY': 'Developing',  # Malaysia
    'MV': 'Developing',  # Maldives
    'ML': 'Developing',  # Mali
    'MR': 'Developing',  # Mauritania
    'MU': 'Developing',  # Mauritius
    'MX': 'Developing',  # Mexico
    'MD': 'Developing',  # Moldova
    'MN': 'Developing',  # Mongolia
    'ME': 'Developing',  # Montenegro
    'MA': 'Developing',  # Morocco
    'MZ': 'Developing',  # Mozambique
    'MM': 'Developing',  # Myanmar
    'NA': 'Developing',  # Namibia
    'NP': 'Developing',  # Nepal
    'NI': 'Developing',  # Nicaragua
    'NE': 'Developing',  # Niger
    'NG': 'Developing',  # Nigeria
    'MK': 'Developing',  # North Macedonia
    'OM': 'Developing',  # Oman
    'PK': 'Developing',  # Pakistan
    'PA': 'Developing',  # Panama
    'PG': 'Developing',  # Papua New Guinea
    'PY': 'Developing',  # Paraguay
    'PE': 'Developing',  # Peru
    'PH': 'Developing',  # Philippines
    'PL': 'Developing',  # Poland
    'QA': 'Developing',  # Qatar
    'RO': 'Developing',  # Romania
    'RU': 'Developing',  # Russia
    'RW': 'Developing',  # Rwanda
    'WS': 'Developing',  # Samoa
    'SA': 'Developing',  # Saudi Arabia
    'SN': 'Developing',  # Senegal
    'RS': 'Developing',  # Serbia
    'SC': 'Developing',  # Seychelles
    'SL': 'Developing',  # Sierra Leone
    'SB': 'Developing',  # Solomon Islands
    'ZA': 'Developing',  # South Africa
    'LK': 'Developing',  # Sri Lanka
    'SD': 'Developing',  # Sudan
    'SR': 'Developing',  # Suriname
    'SZ': 'Developing',  # Eswatini
    'SY': 'Developing',  # Syria
    'TJ': 'Developing',  # Tajikistan
    'TZ': 'Developing',  # Tanzania
    'TH': 'Developing',  # Thailand
    'TL': 'Developing',  # Timor-Leste
    'TG': 'Developing',  # Togo
    'TO': 'Developing',  # Tonga
    'TT': 'Developing',  # Trinidad and Tobago
    'TN': 'Developing',  # Tunisia
    'TR': 'Developing',  # Türkiye
    'TM': 'Developing',  # Turkmenistan
    'UG': 'Developing',  # Uganda
    'UA': 'Developing',  # Ukraine
    'AE': 'Developing',  # United Arab Emirates
    'UY': 'Developing',  # Uruguay
    'UZ': 'Developing',  # Uzbekistan
    'VU': 'Developing',  # Vanuatu
    'VE': 'Developing',  # Venezuela
    'VN': 'Developing',  # Vietnam
    'EH': 'Developing',  # Western Sahara
    'YE': 'Developing',  # Yemen
    'ZM': 'Developing',  # Zambia
    'ZW': 'Developing',  # Zimbabwe
}


def get_s3_cat_cols(df):
    """
    This function searches through the column names of a given DataFrame and 
    returns a list of columns that start with 's3_', contain any characters in 
    between, and end with '_cat' followed by one or more digits.

    Args:
        df (pandas.DataFrame): The input DataFrame from which to extract column names.

    Returns:
        list: A list of column names that match the specified pattern.
    """
    pattern = re.compile(r'^s3_.*_cat\d+$')
    s3_cat_cols = [col for col in df.columns if pattern.match(col)]
    return s3_cat_cols


def get_absolute_emissions_cols(df):
    return ['s1_co2e', 's2_co2e', 's1_and_s2_co2e', 's3_co2e'] + get_s3_cat_cols(df=df) + ['s3_upstream', 's3_downstream', 's3_cat_total']


def get_fundamentals_cols(df):
    mcap_index = df.columns.get_loc("mcap")
    ltdebt_index = df.columns.get_loc("lt_debt")
    fundamental_cols = list(df.columns[mcap_index:ltdebt_index + 1])
    return fundamental_cols


def get_sector_cols():
    return ['econ_sector', 'business_sector', 'industry_group_sector',
            'industry_sector', 'activity_sector', 'econ_sector_code',
            'business_sector_code', 'industry_group_sector_code',
            'industry_sector_code', 'activity_sector_code']


def standardise_missing_values(df, verbose=False):
    """
    Standardises missing values in a DataFrame.

    This function replaces any form of missing values in the DataFrame with `np.nan`.
    It standardizes missing values represented as empty strings or other NA types.

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized missing values.
    """
    missing_before = df.isna().sum().sum()
    
    df = df.applymap(lambda x: np.nan if pd.isna(x) else x)
    df.replace('', np.nan, inplace=True)
    
    missing_after = df.isna().sum().sum()
    changed_count = missing_after - missing_before
    if verbose:
        print("\nSTANDARDISING MISSING VALUES")
        print(f"Number of data points changed to np.nan: {changed_count}")
    return df


def create_new_emissions_aggregates(df, verbose=False):
    """
    Create new emissions aggregates and analyze the proportions of null, zero, and other values in emissions data.

    This function performs the following tasks:
    1. Analyzes the proportions of nulls, zeros, and other values (non-null, non-zero) in the specified emissions columns.
    2. Creates new emissions aggregate columns by combining various upstream, downstream, and overall emissions data.
    3. Optionally prints a wide-format table summarizing the proportions of nulls, zeros, and other values in the emissions data if `verbose` is set to True.
    4. Replaces zero values in the emissions columns with missing values (`np.nan`).

    Args:
        df (pandas.DataFrame): The input DataFrame containing emissions data.
        verbose (bool, optional): If True, prints the proportion of nulls, zeros, and other values for each emissions column. Default is False.

    Returns:
        pandas.DataFrame: The DataFrame with new emissions aggregates and with zero values replaced by missing values in specified columns.
    """
    # Function to calculate proportions of nulls, zeros, and non-null values
    def calculate_proportions(column):
        total_count = len(df)
        null_count = df[column].isna().sum()
        zero_count = (df[column] == 0).sum()
        non_null_count = total_count - null_count
        other_count = non_null_count - zero_count

        null_proportion = null_count / total_count
        zero_proportion = zero_count / total_count
        other_proportion = other_count / total_count

        return [null_proportion, zero_proportion, other_proportion]
       
    s3_cat_cols = get_s3_cat_cols(df)
    upstream_cols = s3_cat_cols[:8]
    downstream_cols = s3_cat_cols[8:]
    emissions_columns = ['s1_co2e', 's2_co2e', 's3_co2e'] + upstream_cols + downstream_cols
    
    proportions_df = pd.DataFrame(columns=emissions_columns, index=['Null', 'Zero', 'Other'])
    if verbose:
        print('\nCREATING NEW EMISSIONS AGGREGATES, SETTING ZEROS TO MISSING')
        # Calculate and store the proportions for each emissions column
        for col in emissions_columns:
            proportions_df[col] = calculate_proportions(col)

        # Print the proportions DataFrame
        print("\nProportions of Nulls, Zeros, and Other Values:")
        print(proportions_df.applymap(lambda x: f"{x:.2%}"))

    # Create some additional emissions aggregates to inspect
    df['s1_and_s2_co2e'] = df['s1_co2e'].fillna(0) + df['s2_co2e'].fillna(0)
    df['s3_upstream'] = df[upstream_cols].sum(axis=1, skipna=True)
    df['s3_downstream'] = df[downstream_cols].sum(axis=1, skipna=True)
    df['s3_cat_total'] = df['s3_upstream'].fillna(0) + df['s3_downstream'].fillna(0)
    
    # Add on the aggregates
    emission_aggregates = ['s1_and_s2_co2e', 's3_upstream', 's3_downstream', 's3_cat_total']
    
    def change_zeros_to_missing(df, cols):
        df[cols] = df[cols].replace(0, np.nan)
        return df
    
    # Replace zeros to missings for aggregates
    df = change_zeros_to_missing(df, cols=emission_aggregates)
    
    return df


def mask_non_reported_co2e(df, verbose=False):
    """
    Masks CO2e values where the co2e method is not 'Reported'.

    This function first sets the co2e_method to either 'Reported' or missing. Next, it modifies the emissions columns 
    to NaN for rows where the relevant co2e method is not 'Reported'.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the c02e method columns and emissions data.

    Returns:
        pandas.DataFrame: The DataFrame with masked CO2e values where the c02e method is not 'Reported'.
    """
    def print_proportions(method_col, df):
        value_counts = df[method_col].value_counts(dropna=False)
        total_count = len(df)
        reported_count = value_counts.get('Reported_value', 0)
        missing_count = total_count - reported_count
        reported_proportion = reported_count / total_count
        missing_proportion = missing_count / total_count

        print(f"\nValue counts for {method_col}:")
        print(value_counts)
        print(f"Proportion 'Reported_value': {reported_proportion:.0%}")
        print(f"Proportion set to missing: {missing_proportion:.0%}")

    if verbose:
        print("MASKING CO2 VALUES THAT ARE NOT REPORTED ENTRIES.")
    s3_cat_cols = get_s3_cat_cols(df)
    s3_upstream_cols = s3_cat_cols[:8]
    s3_downstream_cols = s3_cat_cols[8:]
    method_columns = ['s1_co2e_method', 's2_co2e_method', 's3_upstream_co2e_method', 's3_downstream_co2e_method']

    # Document and mask the CO2e values based on the method columns
    for method_col in method_columns:
        if verbose:
            print("\nMASKING C02 DATA POINTS THAT AREN'T REPORTED")
            print_proportions(method_col, df)


        if method_col == 's3_upstream_co2e_method':
            df.loc[df[method_col] != 'Reported_value', s3_upstream_cols + ['s3_cat_total']] = np.nan
        elif method_col == 's3_downstream_co2e_method':
            df.loc[df[method_col] != 'Reported_value', s3_downstream_cols + ['s3_cat_total']] = np.nan
        elif method_col == 's1_co2e_method':
            df.loc[df[method_col] != 'Reported_value', ['s1_co2e', 's1_and_s2_co2e']] = np.nan
        elif method_col == 's2_co2e_method':
            df.loc[df[method_col] != 'Reported_value', ['s2_co2e', 's1_and_s2_co2e']] = np.nan

    # TODO: Think about how to mask s3_co2e
    return df
    

def mask_co2e_values_not_reported(df):
    
    s3_cat_cols = get_s3_cat_cols(df)
    s3_upstream_cols = s3_cat_cols[:8]
    s3_downstream_cols = s3_cat_cols[8:]

    df.loc[df['s1_co2e_method'] != 'Reported_value', 's1_co2e'] = np.nan
    df.loc[df['s2_co2e_method'] != 'Reported_value', 's2_co2e'] = np.nan
    df.loc[df['s3_upstream_co2e_method'] != 'Reported_value', s3_upstream_cols] = np.nan
    df.loc[df['s3_downstream_co2e_method'] != 'Reported_value', s3_downstream_cols] = np.nan
    
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


def consolidate_observations(df, group_cols, historical_cols, verbose=False):
    """
    Consolidates the observations in the DataFrame if consolidation is possible.

    This function checks if consolidation is possible by ensuring that there are no conflicting values within
    each group. If consolidation is possible, it applies the consolidate_group function to consolidate the data.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.
        group_cols (list): List of columns to group by (e.g., ['instrument', 'year']).
        historical_cols (list): List of historical columns to check for consolidation.

    Returns:
        pd.DataFrame: The consolidated DataFrame if consolidation is possible.
        If consolidation is not possible, returns None and prints the conflicting groups.
    """
    consolidation_not_needed = df.groupby(group_cols).size().all() == 1
    if consolidation_not_needed:
        if verbose:
            print('\nCONSOLIDATING ANY INSTANCES OF MULTIPLE FIRM-YEAR OBSERVATIONS')
            print('Consolidation not required.')
        return df
    
    consolidation, df = consolidation_is_possible(df=df, group_cols=group_cols, historical_cols=historical_cols)
    if consolidation:
        pre_consolidation_length = len(df)
        df_consolidated = df.groupby(group_cols).apply(consolidate_group).reset_index(drop=True)
        post_consolidation_length = len(df_consolidated)
        if verbose:
            print('\nCONSOLIDATING ANY INSTANCES OF MULTIPLE FIRM-YEAR OBSERVATIONS')
            print(f"Consolidation successful, removing {pre_consolidation_length - post_consolidation_length} observations.")
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
        print('\nDROPPING OBSERVATIONS WITH NO FUNDAMENTALS DATA')
        print(f"{n1 - n2} observations dropped because all historical data was missing.")
    
    df = df.reset_index(drop=True)
    return df


def remove_fundamentals_with_low_coverage(df, cols_to_check, threshold=0.1, verbose=False):
    """
    Removes columns from the DataFrame if the proportion of null values exceeds a specified threshold.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols_to_check (list of str): The list of columns to check for null proportions.
        threshold (float): The maximum allowable proportion of null values in a column before it is removed. Default is 0.1 (10%).
        verbose (bool): Whether to print outcome of removing low coverage columns.

    Returns:
        pandas.DataFrame: The DataFrame with columns removed if their null proportion exceeds the threshold.
    """
    if verbose:
            print('\nREVIEWING FUNDAMENTALS COVERAGE AND REMOVING THOSE BELOW THRESHOLD')
    # Iterate over the list of columns and check their null proportions
    for column in cols_to_check:
        null_proportion = df[column].isna().mean()
        if null_proportion > threshold:
            if verbose:
                print(f"Column '{column}' has {null_proportion:.2%} null values and will be removed.")
            df = df.drop(columns=[column])
        else:
            if verbose:
                print(f"Column '{column}' has {null_proportion:.2%} null values and will be retained.")
    return df


def replace_and_fill_zeros_in_fundamentals(df, covariates, verbose=False):
    """
    Replaces zeros in specified covariates with NaN and then fills missing values using forward fill and backward fill.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        covariates (list of str): The list of columns to process.
        verbose (bool, optional): If True, prints the number of data points that were changed. Default is False.

    Returns:
        pandas.DataFrame: The DataFrame with zeros replaced and missing values filled.
    """
    zeros_before = (df[covariates] == 0).sum().sum()
    df[covariates] = df[covariates].replace(0, np.nan)
    nans_after_replace = df[covariates].isna().sum().sum()

    # Fill missing values using forward fill and backward fill
    df[covariates] = df.groupby('instrument', group_keys=False)[covariates].apply(lambda group: group.ffill().bfill())

    nans_after_fill = df[covariates].isna().sum().sum()
    zeros_replaced = nans_after_replace - zeros_before
    nans_filled = nans_after_replace - nans_after_fill

    if verbose:
        print('\nREPLACING ZEROS WITH NULLS IN FUNDAMENTALS AND FILLING THE MISSINGS PER INSTRUMENT')
        print(f"Number of zeros replaced: {zeros_replaced}")
        print(f"Number of NaN values filled: {nans_filled}")

    return df


def drop_dual_listing_duplicates(df, verbose=False):
    """
        Drops the dual-listed instruments for firms with multiple listings, keeping only the instrument with the most data points.

        This function identifies firms that have multiple instruments (dual listings) and retains only the instrument 
        that has the most non-missing data points for a selected subset of columns. The columns checked include both 
        fundamental financial metrics and absolute emissions data.

        Args:
            df (pd.DataFrame): The input DataFrame containing firm data with columns 'firm_name' and 'instrument', 
                            along with other fundamental and emissions-related columns.
            verbose (bool): Whether to print the workings to console.

        Returns:
            pd.DataFrame: A DataFrame with only one instrument per firm, specifically the instrument with the most 
                        non-missing data points in the specified columns. If instruments have the same number of data points, 
                        an arbitrary one is kept.
    """
    n_start = len(df)
    cols_to_audit = get_absolute_emissions_cols(df) + get_fundamentals_cols(df)
    firms_with_multiple_instruments = df.groupby('firm_name')['instrument'].nunique() > 1
    filtered_df = df[df['firm_name'].isin(firms_with_multiple_instruments[firms_with_multiple_instruments].index)]
      
    # Keep only the instrument with the most data points per firm
    instrument_counts = filtered_df.groupby(['firm_name', 'instrument'])[cols_to_audit].apply(lambda x: x.notnull().sum().sum())
    instruments_to_drop = instrument_counts.groupby('firm_name').idxmin().apply(lambda x: x[1]).values
    new_df = df[~df['instrument'].isin(instruments_to_drop)]
    n_end = len(new_df)
    if verbose:
        print('\nCONSOLIDATING DUAL-LISTED FIRMS, KEEPING INSTRUMENT WITH MOST COVERAGE')
        print(f'Dropping {n_start - n_end} rows to not double-count dual-listed firms.')
        print(f'{n_end} rows remaining.')
    return new_df


def map_country_classification(data, country_classification, verbose=False):
    """
    Maps country classification based on country codes, fills missing country codes with HQ country codes,
    and drops the HQ country code column.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the country codes and HQ country codes.
        country_classification (dict): A dictionary mapping country codes to their classifications.
        verbose (bool): Prints to console what's going on.

    Returns:
        pandas.DataFrame: The DataFrame with the country code column filled and classified, and the HQ country code column dropped.
    """
    data['cc'] = data['cc'].fillna(data['cc_hq'])
    data['cc_classification'] = data['cc'].map(country_classification)
    data = data.drop(columns=['cc_hq'])
    
    if verbose:
        print('\nMAPPING COUNTRIES TO DEVELOPED OR EMERGING')
        # Print the number of country classifications added
        added_classifications = data['cc_classification'].notna().sum()
        print(f"Number of country classifications added: {added_classifications}")

        # Print the number of remaining missing classifications
        missing_classifications = data['cc_classification'].isna().sum()
        print(f"Number of remaining missing classifications: {missing_classifications}")
        
    return data


def clean_raw_data_from_load(df,
                             group_cols,
                             historical_cols,
                             year_lower,
                             year_upper,
                             data_coverage_threshold_min,
                             country_classification,
                             verbose=False):

    df = standardise_missing_values(df, verbose=verbose)
    
    # Subset to relevant date range
    financial_years = ['FY' + str(x) for x in range(year_lower, year_upper)]
    df = df[df['financial_year'].isin(financial_years)]
    df['year'] = df['financial_year'].str.replace('FY', '').astype(int)
    if verbose:
        print(f'\nSUBSETTING THE RAW DATA TO THE RANGE {year_lower} to {year_upper-1}')
    print(f'Starting number of observations is {len(df)}')
    # df = mask_non_reported_co2e(df, verbose=verbose)
    df = mask_co2e_values_not_reported(df)
    df = create_new_emissions_aggregates(df, verbose=verbose)
    df = consolidate_observations(df, group_cols=group_cols, historical_cols=historical_cols, verbose=verbose)
    df = drop_rows_with_all_missings(df, fields_to_check=historical_cols, verbose=verbose)
    print(f'Dropping rows with no fundamentals: {len(df)}')
    df = drop_dual_listing_duplicates(df, verbose=verbose)
    print(f'Dropping dual-listed duplicates {len(df)}')
    df = remove_fundamentals_with_low_coverage(df, cols_to_check=historical_cols, threshold=data_coverage_threshold_min, verbose=verbose)
    df = replace_and_fill_zeros_in_fundamentals(df, covariates=get_fundamentals_cols(df), verbose=verbose)
    df = map_country_classification(df, country_classification=country_classification, verbose=verbose)
    return df


def remove_invalid_observations(df, verbose=False):
    # Drop firms in this sector because there are so few
    starting_n = len(df)
    df = df[df['econ_sector'] != 'Academic & Educational Services']
    n_post_econ_sector = len(df)
    
    # Filter out outlier/erroneous values in the fundamentals
    df = df[(df['revenue'].notnull()) & (df['revenue'] > 0)] # Revenue is negative or missing
    n_post_revenue = len(df)
    df = df[(df['intangible_assets'] > 0)] # remove intangible assets < 0
    n_post_intangibles = len(df)
    df = df[(df['capex'] > 0)] # remove capex < 0
    n_post_capex = len(df)
    
    if verbose:
        print('\nREMOVING INVALID VALUES FROM INITIAL DATA LOAD')
        print(f'We begin with {starting_n} observations.')
        print(f'We drop academic and educational service firms as there are so few in their sector. We lose {starting_n - n_post_econ_sector} observations.')
        print(f'We remove {n_post_econ_sector - n_post_revenue} observations with missing or negative revenue.')
        print(f'We remove {n_post_revenue - n_post_intangibles} observations with negative intangibles.')
        print(f'We remove {n_post_revenue - n_post_capex} observations with negative capex.')
        print(f'{n_post_capex} observations remaining.')
        
    return df


def remove_non_normal_observations(df, verbose=False):
    """
    Removes observations with abnormal revenue or CO2e values based on year-over-year ratios for each instrument.

    This function identifies instruments with extreme changes in revenue, Scope 1 and 2 CO2e, Scope 3 upstream CO2e, 
    and Scope 3 downstream CO2e by calculating the ratio of maximum to minimum values over time for each instrument. 
    Observations with ratios exceeding the 99th percentile for any of these metrics are considered abnormal and removed from the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing columns for 'instrument', 'year', 'revenue', 
                               's1_and_s2_co2e', 's3_upstream', and 's3_downstream'.
        verbose (bool, optional): If True, prints the number of observations removed and the number remaining. 
                                  Default is False.

    Returns:
        pandas.DataFrame: The DataFrame with abnormal observations removed.
    """
    # Check for abnormal amounts in revenue or reported co2e values
    df.sort_values(["instrument","year"],ascending=[1,0],inplace=True)
    abnormals = pd.DataFrame(pd.pivot_table(data= df,index = ["instrument"], \
                                values = ["revenue","s1_and_s2_co2e","s3_upstream", "s3_downstream"],aggfunc=[np.min, np.max]).to_records())
    abnormals.columns = ["instrument","min_revenue","min_s12","min_downstream","min_upstream", "max_revenue","max_s12","max_downstream", "max_upstream"]
    abnormals["revenue_ratio"] = abnormals["max_revenue"]/ abnormals["min_revenue"]
    abnormals["ratio_s12"] = abnormals["max_s12"]/ abnormals["min_s12"]
    abnormals["ratio_upstream"] = abnormals["max_upstream"]/ abnormals["min_upstream"]
    abnormals["ratio_downstream"] = abnormals["max_downstream"]/ abnormals["min_downstream"]
    abnormals["ratio_s12"] = abnormals["ratio_s12"].fillna(1)
    abnormals["ratio_upstream"] = abnormals["ratio_upstream"].fillna(1)
    abnormals["ratio_downstream"] = abnormals["ratio_downstream"].fillna(1)
    abnormals['ratio_revenue_abnormal'] = np.where(abnormals["revenue_ratio"]>= np.percentile(abnormals["revenue_ratio"],99), 1, 0)
    abnormals['ratio_s12_abnormal'] = np.where(abnormals["ratio_s12"]>= np.percentile(abnormals["ratio_s12"],99), 1, 0)
    abnormals['ratio_upstream_abnormal'] = np.where(abnormals["ratio_upstream"]>= np.percentile(abnormals["ratio_upstream"],99), 1, 0)
    abnormals['ratio_downstream_abnormal'] = np.where(abnormals["ratio_downstream"]>= np.percentile(abnormals["ratio_downstream"],99), 1, 0)

    df = pd.merge(df, abnormals[['instrument', 'ratio_revenue_abnormal','ratio_s12_abnormal','ratio_upstream_abnormal', 'ratio_downstream_abnormal']], how='left',  on='instrument')
    abnormals = df[(df['ratio_revenue_abnormal']==1) | (df['ratio_s12_abnormal']==1) | (df['ratio_upstream_abnormal']==1) | (df['ratio_downstream_abnormal']==1)]
    abnormal_instruments = abnormals['instrument'].unique()
    pre_drop_n = len(df)
    df = df[~df['instrument'].isin(abnormal_instruments)]
    post_drop_n = len(df)
    if verbose:
        print('\nREMOVING NON-NORMAL REVENUE AND CO2E TRENDS')
        print(f'{pre_drop_n - post_drop_n} observations dropped. {post_drop_n} observations remaining.')
    df = df.drop(columns = ["ratio_revenue_abnormal","ratio_s12_abnormal", "ratio_upstream_abnormal", "ratio_downstream_abnormal"])
    return df
    

def convert_emissions_to_intensities(df):
    """
    Convert Scope 3 emissions to intensities and create new columns.

    This function calculates the intensity of Scope 3 emissions by dividing
    each Scope 3 emissions column by the revenue in millions. It creates new
    columns for each intensity, appending '_intensity' to the original column names.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing emissions data.
                       Expected columns include 'revenue' and various 's3_' columns.

    Returns:
    pd.DataFrame: The DataFrame with new columns for the Scope 3 emissions intensities,
                  where each new column name is the original column name appended with '_intensity'.
    """
    emissions_cols = get_absolute_emissions_cols(df=df)
    
    for col in emissions_cols:
        df[f'{col}_intensity'] = df[col] / (df['revenue'] / 1e6)
    return df


def get_emissions_intensity_cols(df):
    """
    Extract column names from a DataFrame that match a specific pattern related to intensity.

    This function searches through the column names of a given DataFrame and
    returns a list of columns that start with 's3_', contain any characters in
    between, include '_cat' followed by one or more digits, and end with '_intensity'.

    Args:
        df (pandas.DataFrame): The input DataFrame from which to extract column names.

    Returns:
        list: A list of column names that match the specified pattern.
    """
    pattern = re.compile(r'^.*_intensity$')
    s3_cat_intensity_cols = [col for col in df.columns if pattern.match(col)]
    return s3_cat_intensity_cols


def get_s3_cat_intensity_proportion_cols(df):
    """
    Extract column names from a DataFrame that match a specific pattern related to intensity proportion.

    This function searches through the column names of a given DataFrame and
    returns a list of columns that start with 's3_', contain any characters in
    between, include '_cat' followed by one or more digits, and end with '_intensity_proportion'.

    Args:
        df (pandas.DataFrame): The input DataFrame from which to extract column names.

    Returns:
        list: A list of column names that match the specified pattern.
    """
    pattern = re.compile(r'^s3_.*_cat\d+_intensity_proportion$')
    s3_cat_intensity_cols = [col for col in df.columns if pattern.match(col)]
    return s3_cat_intensity_cols


def get_material_s3_category_reporting(data):
    
    # To inspect which categories should be material, run the commented code. 
    # data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    # res = get_s3_cat_rank_per_sector(data)
    # res['sec_code'] = res['sec_code'].astype(str)
    # econ_sectors = data[['econ_sector_code', 'econ_sector']].drop_duplicates()
    # econ_sectors['econ_sector_code'] = econ_sectors['econ_sector_code'].astype(str)
    # combo = res.merge(econ_sectors, how='left', left_on='sec_code', right_on='econ_sector_code')
    # business_sectors = data[['business_sector_code', 'business_sector']].drop_duplicates()
    # business_sectors['business_sector_code'] = business_sectors['business_sector_code'].astype(str)
    # combo = combo.merge(business_sectors, how='left', left_on='sec_code', right_on='business_sector_code')
    # combo['sec_name'] = combo['econ_sector'].combine_first(combo['business_sector'])
    # combo = combo[['sec_code', 'sec_name', 'top_two_categories', 's3_category_importance', 's3_category_values']]
    # combo.to_csv('src/results/materiality.csv', index=False)
    
    # General case: sectors that require non-null values for categories 1 and 11
    general_material = (
        (data['business_sector'].isin([
            'Energy - Fossil Fuels', 'Renewable Energy', # Energy
            'Chemicals', 'Mineral Resources', # Basic Materials
            'Industrial Goods', 'Industrial & Commercial Services', # Industrials
            'Automobiles & Auto Parts', 'Cyclical Consumer Products', 'Retailers', # Consumer Cyclicals
            'Personal & Household Products & Services', 'Food & Drug Retailing', 'Consumer Goods Conglomerates', # Consumer Non-Cyclicals
            'Healthcare Services & Equipment', # Healthcare
            'Technology Equipment', 'Software & IT Services', 'Financial Technology (Fintech) & Infrastructure', # Technology
        ])) &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_use_of_sold_cat11'].notnull())
    )
    
    utility_material = (
        (data['econ_sector'] == 'Utilities') &
        (data['s3_fuel_energy_cat3'].notnull()) &
        (data['s3_use_of_sold_cat11'].notnull())
    )
    
    real_estate_material = ( # Make compatible with FTSE report. Also, need big enough number of unique firms. 
        (data['econ_sector'] == 'Real Estate') &
        (data['s3_capital_goods_cat2'].notnull()) &
        (data['s3_leased_assets_cat13'].notnull())
    )
    
    applied_resources_material = (
        (data['business_sector'] == 'Applied Resources') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_EOL_treatment_cat12'].notnull())
    )

    transportation_material = (
        (data['business_sector'] == 'Transportation') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_fuel_energy_cat3'].notnull())
    )

    cyclical_consumer_services_material = (
        (data['business_sector'] == 'Cyclical Consumer Services') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_franchises_cat14'].notnull())
    )

    food_beverages_material = (
        (data['business_sector'] == 'Food & Beverages') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_transportation_cat4'].notnull())
    )

    pharma_material = (
        (data['business_sector'] == 'Pharmaceuticals & Medical Research') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_capital_goods_cat2'].notnull())
    )

    telecom_material = (
        (data['business_sector'] == 'Telecommunications Services') &
        (data['s3_purchased_goods_cat1'].notnull()) &
        (data['s3_capital_goods_cat2'].notnull())
    )
    
    material_conditions = (
        general_material |
        applied_resources_material |
        transportation_material |
        cyclical_consumer_services_material |
        food_beverages_material |
        pharma_material |
        telecom_material |
        utility_material |
        real_estate_material
    )
    
    return material_conditions

if __name__=="__main__":
    # Load the data
    file_path = 'data/ftse_world_allcap.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    grp_cols = ['instrument', 'financial_year']
    historical_cols = get_fundamentals_cols(data)
    data = clean_raw_data_from_load(data, group_cols=grp_cols, historical_cols=historical_cols,
                                    year_lower=2010, year_upper=2023, data_coverage_threshold_min=0.1,
                                    country_classification=country_classification, verbose=False)

    # Keep observations where at least Scope 1 and Scope 2 data is not missing
    data = data[(data['s1_co2e'].notnull()) & (data['s2_co2e'].notnull())]
    print(f'Removing missing either S1 or S2: {len(data)}')
    # Remove financials
    data = data[data['econ_sector'] != 'Financials']
    print(f'Removing financials: {len(data)}')
    # Drop firms in this sector because there are so few
    data = remove_invalid_observations(data)
    print(f'Firms in sectors with too few peers: {len(data)}')
    # net cash flow is very skewed, let's winsorize
    winsorize(data['net_cash_flow'], limits=(0.005, 0.005), inplace=True)
    # Remove non-normal observations in revenue and co2e
    data = remove_non_normal_observations(data)
    print(f'Removing non-normal observtions: {len(data)}')
    # Get emissions in intensity form
    data = convert_emissions_to_intensities(data)
    
    # Ensure the data is correctly ordered
    data.sort_values(by=['instrument', 'year'], ignore_index=True, inplace=True)
    
    # Create dummy cols for categoricals
    data = pd.concat([data, pd.get_dummies(data['financial_year'],prefix = 'financial_year', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['econ_sector'],prefix = 'econ_sector', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['business_sector'],prefix = 'business_sector', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['industry_group_sector'],prefix = 'industry_group_sector', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['industry_sector'],prefix = 'industry_sector', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['activity_sector'],prefix = 'activity_sector', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['cc'],prefix = 'cc', drop_first=True)], axis=1)
    data = pd.concat([data, pd.get_dummies(data['cc_classification'],prefix = 'cc_classification', drop_first=True)], axis=1)
    data.to_csv('data/ftse_world_allcap_clean.csv', index=False)
    