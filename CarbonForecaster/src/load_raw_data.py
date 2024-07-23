import refinitiv.data as rd
import pandas as pd
import numpy as np
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
        'TR.SEDOL',
        'TR.ICBIndustryCode',
        'TR.ICBIndustry',
        'TR.ICBSupersectorCode',
        'TR.ICBSupersector'
        ],
    'dates':[
        'TR.F.PeriodEndDate',
        'TR.F.SourceDate'
        ],
    'business_metrics':[
        'TR.F.EV', 
        'TR.F.TotRevenue',
        'TR.F.EBIT',
        'TR.F.EmpFTEEquivPrdEnd',
        'TR.F.NetCashFlowOp',
        'TR.F.PPENetTot',
        'TR.F.COSTOFOPREV',
        'TR.F.CAPEXTot',
        'TR.F.IntangTotNet',
        'TR.F.DebtLTTot',
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
    ],
}

static_fields = fields['static']
historical_fields = fields['dates'] + fields['business_metrics'] + fields['industry_production'] + fields['emissions'] + fields['co2e'] + fields['scope3upstream'] + fields['scope3downstream'] 

col_mapping = {
    'Instrument': 'instrument',
    'FY': 'financial_year',
    'Company Common Name': 'firm_name',
    'Country ISO Code of Headquarters': 'cc_hq',
    'SEDOL': 'sedol',
    'ICB Industry code': 'icb_industry_code',
    'ICB Industry name': 'icb_industry_name',
    'ICB Supersector code': 'icb_supersector_code',
    'ICB Supersector name': 'icb_supersector_name',
    'Period End Date': 'period_end_date',
    'Source Filing Date Time': 'source_filing_date',
    'Revenue from Business Activities - Total': 'revenue',
    'Enterprise Value' :'ev',
    'Employees - Full-Time/Full-Time Equivalents - Period End': 'employees',
    'Earnings before Interest & Taxes (EBIT)': 'ebit',
    'Net Cash Flow from Operating Activities': 'net_cash_flow',
    'Property Plant & Equipment - Net - Total': 'net_ppe',
    'Cost of Operating Revenue': 'cost_of_revenue',
    'Intangible Assets - Total - Net': 'intangible_assets',
    'Debt - Long-Term - Total': 'lt_debt',
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


def get_ftse_350():
    """ Quick script to pull the tickers of the FTSE-350"""
    ftse_100_firms = pd.read_csv('data/ftse_350/ftse_100.csv', usecols=['Instrument'])
    ftse_250_firms = pd.read_csv('data/ftse_350/ftse_250.csv', usecols=['Instrument'])
    data = pd.concat([ftse_100_firms, ftse_250_firms])
    data = data['Instrument'].sort_values().tolist()
    return data
    

def load_from_excel(data_dir, static_sheet_name, timeseries_sheet_name):

    # Load and combine 
    static_data = pd.read_excel(data_dir, sheet_name=static_sheet_name)
    static_data.columns.values[0] = 'Instrument'

    historic_data = pd.read_excel(data_dir, sheet_name=timeseries_sheet_name)
    historic_data.columns.values[0] = 'Instrument'
    historic_data.columns.values[1] = 'FY'
    
    data = static_data.merge(historic_data, on='Instrument')
    
    # Apply simple transformations   
    data = rename_columns(data)
    data = coerce_dtypes(data)
    data.rename(columns={'financial_year':'absolute_financial_year'}, inplace=True)
    return data


def call_data_from_api(universe,
                       fields):
    # Define years to loop over 
    fin_yrs = ["FY" + str(i) for i in range(-15, 1)]
    
    rd.open_session()
    static_data = rd.get_data(universe=universe, fields=fields['static'])
    historic_data= list()
    for relative_fin_yr in fin_yrs:
        fy_data= rd.get_data(universe=universe, fields=historical_fields, parameters = {'SDate': relative_fin_yr, 'Curn': 'USD'})
        fy_data['FY'] = relative_fin_yr
        historic_data.append(fy_data)
    data = pd.concat(historic_data, ignore_index=True)
    data = static_data.merge(data, how='outer', on='Instrument')
    data = rename_columns(data)
    data = coerce_dtypes(data)
    data.rename(columns={'financial_year':'relative_financial_year'}, inplace=True)
    rd.close_session()
    return data


def get_fundamentals_in_chunks(universe,
                       fields,
                       chunk_size,
                       filename):
    
    all_data = list()
    for i, chunk in enumerate(chunk_list(universe, chunk_size=chunk_size)):
        logger.info(f'Processing chunk {i+1}')
        try:
            chunk_data = call_data_from_api(universe=universe, fields=fields)
            if chunk_data is not None:
                all_data.append(chunk_data)
                logger.info(f"Successfully loaded and processed chunk {i+1}")
        except Exception as e:
            logger.error(f"An error occurred while processing chunk {i + 1} for firms {chunk}: {e}")
    
    if all_data:    
        data = pd.concat(all_data, ignore_index=True)
        data.drop_duplicates(inplace=True)
        data.to_pickle(filename)
    else:
        print('No data was saved to memory.')
        data = None
    
    
if __name__=="__main__":

    # rd.open_session()
    # test = rd.get_data(universe='3IN.L', fields=['TR.F.PeriodEndDate','TR.F.SourceDate','TR.CO2DirectScope1','TR.CO2IndirectScope2','TR.CO2IndirectScope3','TR.CO2EstimationMethod'], parameters={'SDate':'FY-2'})
    # print(test.loc[test['Source Filing Date Time'] == '2022-05-10 06:15:14', ['CO2 Equivalent Emissions Indirect, Scope 2']])
    # print(test.loc[test['source_filing_date']=='2022-05-10 06:15:14', ['s2_co2e']])
    UNIVERSE = get_ftse_350()
    CHUNK_SIZE = 100
    FILENAME='data/ftse_350/api_call.pkl'
    get_fundamentals_in_chunks(universe=UNIVERSE, fields=fields, chunk_size=CHUNK_SIZE, filename=FILENAME)
