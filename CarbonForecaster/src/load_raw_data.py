import refinitiv.data as rd
import pandas as pd
import re
import logging
import warnings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx INFO messages

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas.core.dtypes.cast")

fields = {
    'static':[
        'TR.CommonName',
        'TR.HQCountryCode',
        'TR.TRBCEconSectorCode',
        'TR.TRBCBusinessSectorCode',
        'TR.TRBCIndustryGroupCode',
        'TR.TRBCIndustryCode',
        'TR.TRBCActivityCode'
        ],
    'size':[
        'TR.F.TotRevenue', 
        'TR.F.EmpFTEEquivPrdEnd',
        'TR.F.MktCap'
        ],
    'co2e':[
        'TR.CO2DirectScope1',
        'TR.CO2IndirectScope2',
        'TR.CO2IndirectScope3',
        'TR.CO2EstimationMethod'
        ],
    'scope3upstream':[
        'TR.UpstreamScope3PurchasedGoodsAndServices',
        'TR.UpstreamScope3CapitalGoods',
        'TR.UpstreamScope3FuelAndEnergy',
        'TR.UpstreamScope3TransportationAndDistribution',
        'TR.UpstreamScope3WasteGeneratedInOperations',
        'TR.UpstreamScope3BusinessTravel',
        'TR.UpstreamScope3EmployeeCommuting',
        'TR.UpstreamScope3LeasedAssets'
        ],
    'scope3downstream':[
        'TR.DownstreamScope3TransportationAndDistribution',
        'TR.DownstreamScope3ProcessingOfSoldProducts',
        'TR.DownstreamScope3UseOfSoldProducts',
        'TR.DownstreamScope3EndOfLifeTreatmentOfSold',
        'TR.DownstreamScope3LeasedAssets',
        'TR.DownstreamScope3Franchises',
        'TR.DownstreamScope3Investments' 
    ]
}

col_mapping = {
    'Instrument': 'instrument',
    'Company Common Name': 'firm_name',
    'Country ISO Code of Headquarters': 'cc_hq',
    'TRBC Economic Sector Code': 'econ_sector_code',
    'TRBC Business Sector Code': 'business_sector_code',
    'TRBC Industry Group Code': 'industry_group_code',
    'TRBC Industry Code': 'industry_code',
    'TRBC Activity Code': 'activity_code',
    'Date': 'date_val',
    'Revenue from Business Activities - Total': 'revenue',
    'Employees - Full-Time/Full-Time Equivalents - Period End': 'employees',
    'Market Capitalization': 'mcap',
    'CO2 Equivalent Emissions Direct, Scope 1': 's1_co2e',
    'CO2 Equivalent Emissions Indirect, Scope 2': 's2_co2e',
    'CO2 Equivalent Emissions Indirect, Scope 3': 's3_co2e',
    'CO2 Estimation Method': 'co2e_method',
    'Upstream scope 3 emissions Purchased goods and services': 's3_purchased_goods_cat1',
    'Upstream scope 3 emissions Capital goods': 's3_capital_goods_cat2',
    'Upstream scope 3 emissions Fuel- and Energy-related Activities': 's3_fuel_energy_cat3',
    'Upstream scope 3 emissions Transportation and Distribution': 's3_transportation_cat4',
    'Upstream scope 3 emissions Waste Generated in Operations': 's3_waste_cat5',
    'Upstream scope 3 emissions Business Travel': 's3_business_travel_cat6',
    'Upstream scope 3 emissions Employee Commuting': 's3_employee_commuting_cat7',
    'Upstream scope 3 emissions Leased Assets': 's3_leased_assets_cat8',
    'Downstream scope 3 emissions Transportation and Distribution': 's3_distribution_cat9',
    'Downstream scope 3 emissions Processing of Sold Products': 's3_processing_products_cat10',
    'Downstream scope 3 emissions Use of Sold Products': 's3_use_of_sold_cat11',
    'Downstream scope 3 emissions End-of-life Treatment of Sold Products': 's3_EOL_treatment_cat12',
    'Downstream scope 3 emissions Leased Assets': 's3_leased_assets_cat13',
    'Downstream scope 3 emissions Franchises': 's3_franchises_cat14',
    'Downstream scope 3 emissions Investments': 's3_investments_cat15'   
}

numeric_cols = [re.compile(r"_code$"), "revenue", "employees", "mcap", re.compile(r"_co2e$"), re.compile(r"^s3")]
string_cols = ['instrument', 'co2e_method']

def is_string_or_single_item_list(obj):
    """
    Checks if the given object is a string or a list with a single item.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a string or a list with a single item, False otherwise.
    """
    return isinstance(obj, str) or (isinstance(obj, list) and len(obj) == 1)


def load_one_firm(universe, fields, interval, start_date, end_date):
    """
    Loads historical data for a single firm.

    Args:
        universe (list or str): The instrument or list containing one instrument.
        fields (list or str): List of fields to fetch.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame with the historical data for the single firm.
    """
    df = rd.get_history(universe=universe, fields=fields, interval=interval, start=start_date, end=end_date).reset_index()
    df['Instrument'] = df.columns.name
    df.columns.name = None
    return df


def load_one_field(universe, fields, interval, start_date, end_date):
    """
    Loads historical data for multiple firms but a single field.

    Args:
        universe (list): List of instruments for which the historical data is to be fetched.
        fields (list or str): A single field to fetch.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame with the historical data for the single field across multiple firms.
    """
    df = rd.get_history(universe=universe, fields=fields, interval=interval, start=start_date, end=end_date).reset_index()
    df = df.melt(id_vars='Date', var_name='Instrument', value_name=df.columns.name)
    df.columns.name = None
    return df


def load_multiple_firms_and_fields(df):
    """
    Reshapes a DataFrame containing historical data from wide format to long format and pivots it to have metrics as separate columns.

    Args:
        df (pandas.DataFrame): The input DataFrame containing historical data.

    Returns:
        pandas.DataFrame: A reshaped DataFrame where each row corresponds to a unique combination of Date and Instrument.
    """
    df = df.reset_index().melt(id_vars=['Date'], var_name=['Instrument', 'Metric'])
    df = df.pivot_table(index=['Date', 'Instrument'], columns='Metric', values='value', aggfunc=lambda x: x).reset_index()
    df.columns.name = None
    return df

def load_historical_data(universe, fields, interval, start_date, end_date):
    """
    Loads and reshapes historical data for a given set of instruments and fields over a specified date range.

    Args:
        universe (list or str): List of instruments or a single instrument for which the historical data is to be fetched.
        fields (dict or list): Dictionary of field types and their respective field names, or a list of field names.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the reshaped historical data for the specified instruments and fields.
    """
    if is_string_or_single_item_list(universe):
        return load_one_firm(universe, fields, interval, start_date, end_date)
    
    if is_string_or_single_item_list(fields):
        return load_one_field(universe, fields, interval, start_date, end_date)
    
    df = None
    
    if isinstance(fields, dict):
        historic_field_types = [field_type for field_type in fields if field_type != 'static']
        
        for historic_set in historic_field_types:
            try:
                data = rd.get_history(universe=universe, fields=fields[historic_set], interval=interval, start=start_date, end=end_date)
                data = load_multiple_firms_and_fields(data)
                if df is None:
                    df = data
                else:
                    df = df.merge(data, on=['Date', 'Instrument'], how='outer')
            except Exception as e:
                logger.error(f"An error occurred while fetching historical data for {historic_set}: {e}")
    else:
        # Handle case where fields is a list
        try:
            data = rd.get_history(universe=universe, fields=fields, interval=interval, start=start_date, end=end_date)
            data = load_multiple_firms_and_fields(data)
            df = data
        except Exception as e:
            logger.error(f"An error occurred while fetching historical data: {e}")
    
    return df

def coerce_dtypes(df, numeric_cols=numeric_cols, string_cols=string_cols):
    """
    Coerces specified columns in a DataFrame to numeric or string format.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        numeric_cols (list): List of column names and regex patterns for columns to be coerced to numeric.
        string_cols (list): List of column names and regex patterns for columns to be coerced to string.

    Returns:
        pandas.DataFrame: The DataFrame with specified columns coerced.
    """
    def is_numeric_col(column):
        for pattern in numeric_cols:
            if isinstance(pattern, str) and column == pattern:
                return True
            elif isinstance(pattern, re.Pattern) and pattern.search(column):
                return True
        return False
    
    def is_string_col(column):
        for pattern in string_cols:
            if isinstance(pattern, str) and column == pattern:
                return True
            elif isinstance(pattern, re.Pattern) and pattern.search(column):
                return True
        return False
    
    numeric_columns = [col for col in df.columns if is_numeric_col(col)]
    string_columns = [col for col in df.columns if is_string_col(col)]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[string_columns] = df[string_columns].astype(str)
    return df

def rename_columns(data, col_mapping=col_mapping):
    """
    Renames columns according to a column mapping dictionary and reorders them.

    Args:
        data (pandas.DataFrame): The input DataFrame.
        col_mapping (dict): Dictionary of old column names as keys and new column names as values.

    Returns:
        pandas.DataFrame: The DataFrame with columns renamed and reordered.
    """
    data.rename(columns=col_mapping, inplace=True)
    data = data[col_mapping.values()]
    return data

def load_all(universe, fields=fields, interval='yearly', start_date='2019-01-01', end_date='2020-01-01'):
    """
    Combines static and historical data loading functions to return the complete dataset in a single call.

    Args:
        universe (list or str): List of instruments or a single instrument for which the data is to be fetched.
        fields (dict): Dictionary of field types and their respective field names.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the complete data for the specified instruments and fields.
    """
    try:
        rd.open_session()
        static_data = rd.get_data(universe=universe, fields=fields['static'])
        historic_data = load_historical_data(universe=universe, fields=fields, interval=interval, start_date=start_date, end_date=end_date)
        data = static_data.merge(historic_data, how='outer', on='Instrument')
        data = rename_columns(data)
        data = coerce_dtypes(data)
        rd.close_session()
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        rd.close_session()
        return None
    return data
