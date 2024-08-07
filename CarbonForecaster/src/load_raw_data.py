import refinitiv.data as rd
import pandas as pd
import numpy as np
import re
import logging
import warnings
import itertools
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
        'TR.CoRPrimaryCountryCode',
        'TR.HQCountryCode',
        'TR.TRBCEconomicSector',
        'TR.TRBCBusinessSector',
        'TR.TRBCIndustryGroup',
        'TR.TRBCIndustry',
        'TR.TRBCActivity',
        'TR.TRBCEconSectorCode',
        'TR.TRBCBusinessSectorCode',
        'TR.TRBCIndustryGroupCode',
        'TR.TRBCIndustryCode',
        'TR.TRBCActivityCode'
        ],
    'business_metrics':[
        'TR.F.TotRevenue',
        'TR.F.EBIT',
        'TR.F.EBITDA',
        'TR.F.GrossProfIndPropTot',
        'TR.F.NetCashFlowOp',
        'TR.F.TotAssets',
        'TR.F.TotCurrAssets',
        'TR.F.TotCurrLiab',
        'TR.F.InvntTot',
        'TR.F.LoansRcvblTot',
        'TR.F.PPEGrossTot',
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
        'TR.Scope1EstMethod',
        'TR.Scope2EstMethod',
        'TR.Scope3EstDownstreamMethod',
        'TR.Scope3EstUpstreamMethod',
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


col_mapping = {
    'Instrument': 'instrument',
    'Financial Period Absolute': 'financial_year',
    'Company Common Name': 'firm_name',
    'ISO2 Code of Primary Country of Risk': 'cc',
    'Country ISO Code of Headquarters': 'cc_hq',
    'TRBC Economic Sector Name': 'econ_sector',
    'TRBC Business Sector Name': 'business_sector',
    'TRBC Industry Group Name': 'industry_group_sector',
    'TRBC Industry Name': 'industry_sector',
    'TRBC Activity Name': 'activity_sector',
    'TRBC Economic Sector Code': 'econ_sector_code',
    'TRBC Business Sector Code': 'business_sector_code',
    'TRBC Industry Group Code': 'industry_group_sector_code',
    'TRBC Industry Code': 'industry_sector_code',
    'TRBC Activity Code': 'activity_sector_code',
    'Company Market Cap': 'mcap',
    'Revenue from Business Activities - Total': 'revenue',
    # 'Number of Employees': 'employees',
    'Earnings before Interest & Taxes (EBIT)': 'ebit',
    'Earnings before Interest Taxes Depreciation & Amortization': 'ebitda',
    'Gross Profit - Industrials/Property - Total': 'gross_profit',
    'Net Cash Flow from Operating Activities': 'net_cash_flow',
    'Total Assets': 'assets',
    'Total Current Assets': 'current_assets',
    'Total Current Liabilities': 'current_liabilities',
    'Inventories - Total': 'inventories',
    'Loans & Receivables - Total': 'receivables',
    'Property Plant & Equipment - Gross - Total': 'gross_ppe',
    'Property Plant & Equipment - Net - Total': 'net_ppe',
    'Cost of Operating Revenue': 'cost_of_revenue',
    'Capital Expenditures - Total': 'capex',
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
    'Scope 1 Estimated Method': 's1_co2e_method',
    'Scope 2 Estimated Method': 's2_co2e_method',
    'Scope3 Est Downstream Method': 's3_downstream_co2e_method',
    'Scope3 Est Upstream Method': 's3_upstream_co2e_method',
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


def get_ftse_all_cap_universe():
    ftse_all_cap = pd.read_csv('data/FTSE_AllWorld_20221231.csv', usecols=['Constituent RIC']).dropna().drop_duplicates().values.tolist()
    ftse_all_cap = [item[0] for item in ftse_all_cap] 
    return ftse_all_cap


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
    data.rename(columns={'financial_year':'absolute_financial_year'}, inplace=True)
    return data


def get_non_fundamentals(universe, fields, financial_years, parameters):
    
    data = list(itertools.product(universe, financial_years))
    data = pd.DataFrame(data, columns = ['Instrument', 'Financial Period Absolute'])
    year_end_dates = pd.to_datetime([str(year) + '-12-31' for year in range(2010, 2025, 1)])

    def find_closest_date(date, date_list):
        closest_date = date_list[np.argmin(np.abs(date_list - date))]
        return closest_date
    
    def find_previous_year_end(date, date_list):
        # Filter the list to only include dates before the current date
        previous_dates = date_list[date_list < date]
        # If there are no earlier dates, return the earliest available date
        if len(previous_dates) == 0:
            return date_list.min()
        # Return the maximum date from the filtered list (i.e., the most recent previous year-end)
        return previous_dates.max()

    for field in fields:
        rd.open_session()
        if field == 'TR.CompanyMarketCap':
            flds = ['TR.CompanyMarketCap', 'TR.CompanyMarketCap.calcdate'] 
        elif field == 'TR.CompanyNumEmploy':
            flds = ['TR.CompanyNumEmploy', 'TR.CompanyNumEmployDate']
        print(f"Loading data for {field}")
        historical_data = list()
        for chunk in chunk_list(lst=universe, chunk_size=500):
            temp = rd.get_data(universe=chunk, fields=flds, parameters=parameters)
            historical_data.append(temp)
        historical_data = pd.concat(historical_data, ignore_index=True)
        historical_data.columns.values[2] = 'Financial Period Absolute'
        historical_data['Financial Period Absolute'] = pd.to_datetime(historical_data['Financial Period Absolute'])
        historical_data = historical_data.dropna()
        # historical_data['year'] = historical_data['Financial Period Absolute'].dt.year
        # If the years are unique, i.e. one observation per year, keep it, otherwise, round to nearest year
        def process_group(group):
            if len(group['Financial Period Absolute'].dt.year.unique()) == len(group['Financial Period Absolute'].dt.year):
                group['Financial Period Absolute'] = group['Financial Period Absolute']
            else:
                group['Financial Period Absolute'] = group['Financial Period Absolute'].apply(lambda x: find_previous_year_end(x, year_end_dates))
            return group
        historical_data = historical_data.groupby('Instrument', group_keys=False).apply(process_group)
        historical_data['Financial Period Absolute'] = historical_data['Financial Period Absolute'].dt.year.astype(int)
        historical_data['Financial Period Absolute'] = 'FY' + historical_data['Financial Period Absolute'].astype(str)
        data = data.merge(historical_data, on = ['Instrument', 'Financial Period Absolute'], how = 'outer')
        rd.close_session()
    return data


def save_raw_data_from_api(file_dir, universe, fields, financial_years, parameters):
    static_fields = fields['static']
    historical_fields = fields['business_metrics'] + fields['industry_production'] + fields['emissions'] + fields['co2e'] + fields['scope3upstream'] + fields['scope3downstream'] 
    
    rd.open_session()
    
    # Create firm-year combinations
    all_potential_observations = list(itertools.product(universe, financial_years))
    all_potential_observations = pd.DataFrame(all_potential_observations, columns = ['Instrument', 'Financial Period Absolute'])
    
    # Load the static data e.g. company name, HQ country, sector
    static_data = list()
    for chunk in chunk_list(lst=universe, chunk_size=500):
        data = rd.get_data(universe=chunk, fields=static_fields)
        static_data.append(data)
    static_data = pd.concat(static_data, ignore_index=True)
    
    # The base dataframe 
    data = all_potential_observations.merge(static_data, on='Instrument', how='outer')
    
    rd.close_session()
    
    # Get market cap data
    non_fundamentals = get_non_fundamentals(universe=universe, fields = ['TR.CompanyMarketCap'], financial_years=financial_years, parameters=parameters) # 'TR.CompanyNumEmploy'
    data = data.merge(non_fundamentals, on = ['Instrument', 'Financial Period Absolute'], how = 'outer')
    
    # Load the historical variables one at a time, and join on using their fperiod
    for field in historical_fields:
        rd.open_session()
        flds = [field, field + '.fperiod'] 
        print(f"Loading data for {field}")
        historical_data = list()
        for chunk in chunk_list(lst=universe, chunk_size=500):
            temp = rd.get_data(universe=chunk, fields=flds, parameters=parameters)
            historical_data.append(temp)
        historical_data = pd.concat(historical_data, ignore_index=True)
    
        data = data.merge(historical_data, on = ['Instrument', 'Financial Period Absolute'], how = 'outer')
        rd.close_session()
        
    # Rename columns and drop any duplicates
    data = rename_columns(data)
    data.drop_duplicates(inplace=True) 
    
    # Save
    data.to_pickle(file_dir)
    print(f"Data saved to {file_dir}")
    
    
if __name__=="__main__":
    universe = get_ftse_all_cap_universe()
    financial_years = ['FY' + str(i) for i in range(2010, 2024, 1)]
    params = {"SDate" :0 , "EDate" :-14, "FRQ":"FY", "Curn": "USD"}
    save_raw_data_from_api('data/ftse_world_allcap.pkl',
                           universe=universe,
                           fields=fields,
                           financial_years=financial_years,
                           parameters=params)
    