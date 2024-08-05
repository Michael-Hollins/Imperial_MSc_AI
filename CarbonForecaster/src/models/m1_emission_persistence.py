# Predictive model with fixed effects

# Rationale: If there is high persistence in emissions, then we can justify using 
# subsequently reported data as a proxy for emissions. The inspiration for this 
# approach comes from Kalesnik et al (2022) "Do Corporate Carbon Emissions Data 
# Enable Investors to Mitigate Climate Change?"

import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from clean_raw_data import get_absolute_emissions_cols
from clean_raw_data import get_sector_cols


# Functions
def predictive_regression_fixed_effects(df, endogenous_var, industry_or_sector_dummies, country_dummy_cols, year_dummy_cols):
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


# Define parameters and relevant variables
data_load_path = './data/ftse_world_allcap_clean.csv'

# Load in the cleaned data
data = pd.read_csv(data_load_path)

# Log-transform the emissions variables due to the skewness, and then create the lagged value
data.set_index(['instrument', 'year'], inplace=True) # Set 'instrument' as the index to create lagged values
emissions_cols = get_absolute_emissions_cols(data)
for col in emissions_cols:
    mask = (data[col] > 0) & data[col].notnull()
    data[f'{col}_log'] = np.nan  # Initialize with NaNs
    data.loc[mask, f'{col}_log'] = np.log(data.loc[mask, col])
    data[f'{col}_log_lag'] = data.groupby(level='instrument')[f'{col}_log'].shift(1)

# Reset index to include 'instrument' and 'year' as columns again
data.reset_index(inplace=True)

# Convert categorical variables to dummy variables
categorical_cols = ['econ_sector_code', 'cc', 'year']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
econ_sector_dummy_cols = [x for x in list(data.columns) if re.match('^econ_sector_code_\d{2}$', x)]
country_dummy_cols = [x for x in list(data.columns) if re.match('^cc_[A-Z]{2}$', x)]
year_dummy_cols = [x for x in list(data.columns) if re.match('^year_\d{4}$', x)]

if __name__=='__main__':
    results = list()
    for col in emissions_cols:
        model = predictive_regression_fixed_effects(data, 
                                                    endogenous_var=col, 
                                                    industry_or_sector_dummies=econ_sector_dummy_cols,
                                                    country_dummy_cols=country_dummy_cols,
                                                    year_dummy_cols=year_dummy_cols)
        results.append(model_dict_summary(model))
        
    res = pd.DataFrame(results)
    print(res)
