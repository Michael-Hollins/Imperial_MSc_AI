# Load in the raw data from Refinitiv
import refinitiv.data as rd
import pandas as pd
import re
rd.open_session()

def reshape_historic_data(df):
    """
    Reshapes a DataFrame containing historic data from wide format to long format,
    and then pivots it to have metrics as separate columns.

    Args:
        df (pandas.DataFrame): The input DataFrame containing historical data. It can have a single-level
                               or multi-level column index. If it has a multi-level column index, the first
                               level represents the instruments and the second level represents the metrics.
                               The 'Date' column should be present as well.

    Returns:
        pandas.DataFrame: A reshaped DataFrame where each row corresponds to a unique combination of
                          Date and Instrument. If the input DataFrame has multiple metrics, each metric
                          is a separate column. If there is only one metric, it will be in a column named
                          after the original column name.

    Example:
        Given a DataFrame 'df' with the following structure:
        
            Date        Instrument1          Instrument2          ...
                        Metric1  Metric2     Metric1  Metric2     ...
            2014-12-31  ...      ...         ...      ...         ...
            2015-12-31  ...      ...         ...      ...         ...
            ...

        The function returns a DataFrame with the following structure:
        
            Date        Instrument  Metric1  Metric2  ...
            2014-12-31  ...         ...      ...      ...
            2015-12-31  ...         ...      ...      ...
            ...
    """
    if df.columns.nlevels == 1:
        col_name = df.columns.name
        df = df.reset_index().melt(id_vars=['Date'], var_name=['Instrument'], value_name=col_name)
    elif df.columns.nlevels > 1:
        df = df.reset_index().melt(id_vars=['Date'], var_name=['Instrument', 'Metric'])
        df = df.pivot_table(index=['Date', 'Instrument'], columns = 'Metric', values='value', aggfunc=lambda x: x).reset_index()
    df.columns.name = None
    return df

def load_historical_data(universe, fields, interval, start_date, end_date):
    """
    Loads and reshapes historical data for a given set of instruments and fields over a specified date range.

    Args:
        universe (list): List of instruments for which the historical data is to be fetched.
        fields (dict): Dictionary where keys are field types (e.g., 'co2e', 'price', etc.) and values are lists of field names to fetch.
        interval (str): Time interval for the data (e.g., 'daily', 'monthly', 'yearly').
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the reshaped historical data for the specified instruments and fields, merged on 'Date' and 'Instrument'.

    Example:
        universe = ['XOM', 'CVX']
        fields = {'price': ['CLOSE_CF', 'OPEN_PRC'], 'co2e': ['TR.CO2DirectScope1']}
        interval = 'yearly'
        start_date = '2020-01-01'
        end_date = '2022-12-31'

        df = load_historical_data(universe, fields, interval, start_date, end_date)
    """
    df = None
    historic_field_types = [i for i in fields if i != 'static']
    
    for historic_set in historic_field_types:
        data = rd.get_history(universe=universe, fields=fields[historic_set], interval=interval, start=start_date, end=end_date)
        data = reshape_historic_data(data)
        if df is None:
            df = data
        else:
            df = df.merge(data, on=['Date', 'Instrument'], how='outer')
    
    return df

def coerce_dtypes(df, numeric_cols, string_cols):
    """
    Coerces specified columns in a pd.DataFrame to numeric or stringformat.

    Args:
        df (pandas.DataFrame): The input DataFrame
        numeric_cols (list): List of column names and regex patterns for columns to be coerced.
        string_cols (list): List of column names and regex patterns for columns to be coerced.
        
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


# Constants
LARGEST_MINING_COMPANIES=['2222.SE', 'XOM', 'CVX', 'SHEL.L', '601857.SS', 'TTEF.PA', 'GAZP.MM', 'COP', 'BP.L', 'ROSN.MM']
INTERVAL='1Y'
START_DATE='2015-01-01'
END_DATE='2024-01-01'


static_data = rd.get_data(universe=LARGEST_MINING_COMPANIES, fields = fields['static'])    
historic_data = load_historical_data(universe=LARGEST_MINING_COMPANIES, fields=fields, interval=INTERVAL, start_date=START_DATE, end_date=END_DATE)
data = static_data.merge(historic_data, how='outer', on='Instrument')
data.rename(columns=col_mapping, inplace=True)
data = data[col_mapping.values()]
#numeric_cols = [re.compile(r"_code$"), "revenue", "employees", "mcap", re.compile(r"_co2e$"), re.compile(r"^s3")]
#string_cols = ['instrument', 'co2e_method']
#data = coerce_dtypes(data, numeric_cols, string_cols)

rd.close_session()