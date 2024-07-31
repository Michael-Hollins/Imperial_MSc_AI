
import numpy as np
import pandas as pd
from load_raw_data import load_from_excel
from load_raw_data import col_mapping
import pickle
import re

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
    s3_cat_cols = get_s3_cat_cols(df)
    df.loc[df['co2e_method'] != 'Reported', ['s1_co2e', 's2_co2e', 's3_co2e',
                                             'policy_emissions_score', 'target_emissions_score', 'emissions_trading_score'] + s3_cat_cols] = np.nan
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


def convert_s3_emissions_to_intensities(df):
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
    s3_cat_cols = get_s3_cat_cols(df)
    
    for col in s3_cat_cols:
        df[f'{col}_intensity'] = df[col] / (df['revenue'] / 1e6)
    return df


def get_s3_cat_intensity_cols(df):
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
    pattern = re.compile(r'^s3_.*_cat\d+_intensity$')
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


def get_sector_cols():
    return ['icb_industry_code', 'icb_industry_name', 'icb_supersector_code', 'icb_supersector_name', 'icb_sector_code', 'icb_sector_name']

if __name__=="__main__":
    # Load the data
    file_path = 'data/ftse_global_allcap.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Basic cleaning
    grp_cols = ['instrument', 'financial_year']
    revenue_index = data.columns.get_loc("revenue")
    historic_cols = list(data.columns[revenue_index:])
    data = clean_raw_data_from_load(data, group_cols=grp_cols, historical_cols=historic_cols)
    
    # Add some more variables and subset to a relevant date period
    data['year'] = data['financial_year'].str.replace('FY', '').astype(int)
    data = data[(data['year'] >= 2016) & (data['year'] <= 2023)]
    data['cc_classification'] = data['cc'].map(country_classification)
    
    # Get emissions in intensity form
    data = convert_s3_emissions_to_intensities(data)
    
    # Save
    data.to_csv('data/ftse_global_allcap_clean.csv', index=False)
