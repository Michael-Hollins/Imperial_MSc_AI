# Predictive model with fixed effects

# Rationale: If there is high persistence in emissions, then we can justify using 
# subsequently reported data as a proxy for emissions. The inspiration for this 
# approach comes from Kalesnik et al (2022) "Do Corporate Carbon Emissions Data 
# Enable Investors to Mitigate Climate Change?"

import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from clean_raw_data import get_s3_cat_cols

# Define parameters and relevant variables
data_load_path = './data/ftse_global_allcap_clean.csv'

# Load in the cleaned data
data = pd.read_csv(data_load_path)

# Create some additional emissions aggregates to inspect
data['s1_and_s2_co2e'] = data['s1_co2e'] + data['s2_co2e']
upstream_cols = ['s3_purchased_goods_cat1', 's3_capital_goods_cat2',
       's3_fuel_energy_cat3', 's3_transportation_cat4', 's3_waste_cat5',
       's3_business_travel_cat6', 's3_employee_commuting_cat7',
       's3_leased_assets_cat8']
downstream_cols = ['s3_distribution_cat9',
       's3_processing_products_cat10', 's3_use_of_sold_cat11',
       's3_EOL_treatment_cat12', 's3_leased_assets_cat13',
       's3_franchises_cat14', 's3_investments_cat15']
data['s3_upstream'] = data[upstream_cols].sum(axis=1)
data['s3_downstream'] = data[downstream_cols].sum(axis=1)
data['s3_cat_total'] = data['s3_upstream'] + data['s3_downstream']

# Ensure the data is correctly ordered
data.sort_values(by=['instrument', 'year'], ignore_index=True, inplace=True)

# Log-transform the emissions variables due to the skewness, and then create the lagged value
data.set_index(['instrument', 'year'], inplace=True) # Set 'instrument' as the index to create lagged values
emissions_cols = ['s1_co2e', 's2_co2e', 's1_and_s2_co2e',
                  's3_co2e', 's3_upstream', 's3_downstream',
                  's3_cat_total'] + get_s3_cat_cols(data)
for col in emissions_cols:
    mask = (data[col] > 0) & data[col].notnull()
    data[f'{col}_log'] = np.nan  # Initialize with NaNs
    data.loc[mask, f'{col}_log'] = np.log(data.loc[mask, col])
    data[f'{col}_log_lag'] = data.groupby(level='instrument')[f'{col}_log'].shift(1)

# Reset index to include 'instrument' and 'year' as columns again
data.reset_index(inplace=True)

# Convert categorical variables to dummy variables
categorical_cols = ['icb_industry_code', 'icb_supersector_code', 'icb_sector_code', 'cc', 'year']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
industry_dummy_cols = [x for x in list(data.columns) if re.match('^icb_industry_code_\d{2}\.0$', x)]
supersector_dummy_cols = [x for x in list(data.columns) if re.match('^icb_supersector_code_\d{4}\.0$', x)]
sector_dummy_cols = [x for x in list(data.columns) if re.match('^icb_sector_code_\d{6}\.0$', x)]
country_dummy_cols = [x for x in list(data.columns) if re.match('^cc_[A-Z]{2}', x)]
year_dummy_cols = [x for x in list(data.columns) if re.match('^year_\d{4}', x)]

def predictive_regression_fixed_effects(df, endogenous_var='s1_co2e', industry_or_sector_dummies=industry_dummy_cols):
    fixed_effects = industry_or_sector_dummies + country_dummy_cols + year_dummy_cols
    y_var = endogenous_var + '_log'
    y_var_lag = y_var + '_lag'
    df = df[['instrument', y_var, y_var_lag] + fixed_effects].dropna()

    # Prepare data for regression
    X = df.drop(columns=['instrument', y_var])
    X = sm.add_constant(X)
    y = df[y_var]

    # Model
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['instrument']})
    return model


def model_dict_summary(model):   
    return {'N': int(model.nobs),
        'adjusted_r_squared':round(model.rsquared_adj, 3),
        'x1': model.params.index[1].replace('_log_lag', ''), 
        'beta_1':round(model.params[model.params.index[1]],3),
        'beta_1_p_val':model.pvalues[model.params.index[1]]}

results = list()
for col in emissions_cols:
    model = predictive_regression_fixed_effects(data, endogenous_var=col)
    results.append(model_dict_summary(model))
    
res = pd.DataFrame(results)

# How predictable are changes to emissions?

# Convert each observation to a % change
# Take the lag
# Model as above