import refinitiv.data as rd
import pandas as pd
import re
import logging
import warnings
import pickle

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
    'business_metrics':[
        'TR.F.EV', 
        'TR.F.MktCap',
        'TR.F.TotRevenue',
        'TR.F.EBIT',
        'TR.F.EmpFTEEquivPrdEnd',
        'TR.F.NetCashFlowOp',
        'TR.F.PPENetTot',
        'TR.COGSActValue',
        'TR.F.CAPEXTot',
        'TR.F.IntangTotNet',
        'TR.F.LTDebtExclCapLease',
        ],
    'industry_production':[
        'TR.F.CrudeOilProdTot',
        'TR.F.GasLiquidsProdTot',
        'TR.F.NatGasProdTot',
        'TR.WasteTotal',
    ],
    'emissions':[
        'TR.PolicyEmissionsScore',
        'TR.TargetEmissionsScore',
        'TR.EmissionsTradingScore',
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
    'Enterprise Value' :'ev',
    'Employees - Full-Time/Full-Time Equivalents - Period End': 'employees',
    'Market Capitalization': 'mcap',
    'Earnings before Interest & Taxes (EBIT)': 'ebit',
    'Net Cash Flow from Operating Activities': 'net_cash_flow',
    'Property Plant & Equipment - Net - Total': 'net_ppe',
    'Cost Of Goods Sold - Actual': 'cogs',
    'Intangible Assets - Total - Net': 'intangible_assets',
    'Long-Term Debt excluding Capitalized Leases': 'lt_debt',
    'Crude Oil - Production - Total': 'prod_crude_oil',
    'Gas Liquids (NGL) - Production - Total': 'prod_natural_gas_liquids',
    'Natural Gas - Production - Total': 'prod_nat_gas',
    'Waste Total': 'waste',
    'Policy Emissions Score': 'policy_emissions_score',
    'Targets Emissions Score': 'target_emissions_score',
    'Emissions Trading Score': 'emissions_trading_score',
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

numeric_cols = [re.compile(r"_code$"), "revenue", "ev", "employees", "mcap", re.compile(r"_co2e$"), re.compile(r"^s3")]
string_cols = ['instrument', 'co2e_method']

def is_string_or_single_item_list(obj):
    """
    Checks if the given object is a string or a list with a single string item.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a string or a list with a single item, False otherwise.
    """
    return isinstance(obj, str) or (isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], str))


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
    df = None
    
    if isinstance(fields, dict):
        historic_field_types = [field_type for field_type in fields if field_type != 'static']
        
        for historic_set in historic_field_types:
            try:
                data = rd.get_history(universe=universe, fields=fields[historic_set], interval=interval, start=start_date, end=end_date).reset_index()
                data['Instrument'] = data.columns.name
                data.columns.name = None
                if df is None:
                    df = data
                else:
                    df = df.merge(data, on=['Date', 'Instrument'], how='outer')
            except Exception as e:
                logger.error(f"An error occurred while fetching historical data for {historic_set}: {e}")
    else:
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
    df = None

    if isinstance(fields, dict):
        historic_field_types = [field_type for field_type in fields if field_type != 'static']

        for historic_set in historic_field_types:
            try:
                data = rd.get_history(universe=universe, fields=fields[historic_set], interval=interval, start=start_date, end=end_date).reset_index()
                data = data.melt(id_vars='Date', var_name='Instrument', value_name=data.columns.name)
                data.columns.name = None
                if df is None:
                    df = data
                else:
                    df = df.merge(data, on=['Date', 'Instrument'], how='outer')
            except Exception as e:
                logger.error(f"An error occurred while fetching historical data for {historic_set}: {e}")
    else:
        df = rd.get_history(universe=universe, fields=fields, interval=interval, start=start_date, end=end_date).reset_index()
        df = df.melt(id_vars='Date', var_name='Instrument', value_name=df.columns.name)
        df.columns.name = None

    return df


def reshape_multiple_firms_and_fields(df):
    """
    Reshapes a DataFrame containing historical data from long format to wide format by pivoting it so that each metric becomes a separate column. 
    It ensures that for each combination of Date and Instrument, the non-missing value is retained, or if both are missing, the first occurrence is kept.

    Args:
        df (pandas.DataFrame): The input DataFrame containing historical data in long format with columns 'Date', 'Instrument', 'Metric', and 'value'.

    Returns:
        pandas.DataFrame: A reshaped DataFrame where each row corresponds to a unique combination of Date and Instrument, and each metric is a separate column.
    """
    try:
        df = df.reset_index().melt(id_vars=['Date'], var_name=['Instrument', 'Metric'])
        df = df.sort_values(by=['Date', 'Instrument', 'Metric', 'value'], na_position='last')
        df = df.drop_duplicates(subset=['Date', 'Instrument', 'Metric'], keep='first')
        df = df.pivot(index=['Date', 'Instrument'], columns='Metric', values='value').reset_index()
        df.columns.name = None
        return df
    except Exception as e:
        logger.error(f"An error occurred during reshaping: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

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
                data = reshape_multiple_firms_and_fields(data)
                if df is None:
                    df = data
                else:
                    df = df.merge(data, on=['Date', 'Instrument'], how='outer', suffixes=('', f'_{historic_set}'))
            except Exception as e:
                logger.error(f"An error occurred while fetching historical data for {historic_set}: {e}")
    
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

def load_all(universe, fields=fields, interval='yearly', start_date='2019-01-01', end_date='2020-01-01', debug_mode=False):
    """
    Combines static and historical data loading functions to return the complete dataset in a single call.

    Args:
        universe (list or str): List of instruments or a single instrument for which the data is to be fetched.
        fields (dict): Dictionary of field types and their respective field names.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.
        debug_mode (bool): Flag to print logs of what successfully loads when.

    Returns:
        pandas.DataFrame: A DataFrame containing the complete data for the specified instruments and fields.
    """
    try:
        rd.open_session()
        if debug_mode:
            logger.info(f"Loading static data for universe: {universe}")
        static_data = rd.get_data(universe=universe, fields=fields['static'])
        if debug_mode:
            logger.info(f"Loading historical data for universe: {universe}")
        historic_data = load_historical_data(universe=universe, fields=fields, interval=interval, start_date=start_date, end_date=end_date)
        if debug_mode:
            logger.info(f"Combining static and historical data for universe: {universe}")
        data = static_data.merge(historic_data, how='outer', on='Instrument')
        data = rename_columns(data)
        data = coerce_dtypes(data)
        data['year'] = data['date_val'].dt.year
    except Exception as e:
        logger.error(f"An error occurred while loading data for universe: {universe}: {e}")
        data = None
    finally:
         rd.close_session()
    return data


def chunk_list(lst, chunk_size):
    """
    Splits the input list into chunks of the specified size

    Args:
        lst (list): The full list to be split
        chunk_size (int): The size of each chunk
        
    Output:
        list: The next chunk of the input list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_mock_universe():
    """ Quick script to pull in a selection of firms from Refinitiv"""
    eu_growth = pd.read_excel('data/mock_universe/eu_large_growth_loans_12M.xlsx', usecols=['Issuer/Borrower PermID', 'Issuer/Borrower Name Full'])
    eu_value = pd.read_excel('data/mock_universe/eu_large_value_loans_12M.xlsx', usecols=['Issuer/Borrower PermID', 'Issuer/Borrower Name Full'])
    us_firms = pd.read_excel('data/mock_universe/us_firms_loans_12M.xlsx', usecols=['Issuer/Borrower PermID', 'Issuer/Borrower Name Full'])
    data = pd.concat([eu_growth, eu_value, us_firms])
    data.columns = ['permID', 'name']
    data['permID'] = pd.to_numeric(data['permID'], errors='coerce')
    data.dropna(inplace=True)
    data['permID'] = data['permID'].astype('Int64').astype(str)
    data = data['permID'].drop_duplicates().tolist()
    return data


def get_ftse_350():
    """ Quick script to pull the tickers of the FTSE-350"""
    ftse_100_firms = pd.read_csv('data/ftse_350/ftse_100.csv', usecols=['Instrument'])
    ftse_250_firms = pd.read_csv('data/ftse_350/ftse_250.csv', usecols=['Instrument'])
    data = pd.concat([ftse_100_firms, ftse_250_firms])
    data = data['Instrument'].tolist()
    return data


def save_mock_universe():
    MOCK_UNIVERSE = get_ftse_350()
    INTERVAL = '1Y'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    CHUNK_SIZE = 50

    all_data = list()
    for i, chunk in enumerate(chunk_list(MOCK_UNIVERSE , CHUNK_SIZE)):
        logger.info(f'Processing chunk {i+1}')
        try:
            chunk_data = load_all(chunk, interval=INTERVAL, start_date=START_DATE, end_date=END_DATE, debug_mode=True)
            if chunk_data is not None:
                all_data.append(chunk_data)
                logger.info(f"Successfully loaded and processed chunk {i+1}")
        except Exception as e:
            logger.error(f"An error occurred while processing chunk {i + 1} for firms {chunk}: {e}")
    
    if all_data:    
        data = pd.concat(all_data, ignore_index=True)        
        print('Data load complete.')
        data.to_pickle('data/mock_universe/mock_universe.pkl')
    else:
        print('No data was saved to memory.')
    
if __name__=="__main__":
    save_mock_universe()