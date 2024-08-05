# Naive model: sector medians per year

import pandas as pd
from clean_raw_data import get_absolute_emissions_cols
from clean_raw_data import get_emissions_intensity_cols
from clean_raw_data import change_zeros_to_missing

# Load in the data
data_load_path = './data/ftse_global_allcap_clean.csv'
data = pd.read_csv(data_load_path)

# Change emissions cols to missing rather than zero for modelling
emissions_absolute_cols = get_absolute_emissions_cols(data)
emissions_intensity_cols = get_emissions_intensity_cols(data)
emissions_cols = emissions_absolute_cols + emissions_intensity_cols
data = change_zeros_to_missing(data, emissions_cols)

# Get first non-missing value per emissions intensity per instrument
first_entries = data[['instrument', 'icb_industry_code', 'year'] + emissions_intensity_cols].melt(id_vars=['instrument', 'icb_industry_code', 'year'], var_name='emission_intensity_col').dropna().reset_index(drop=True)
first_entries = first_entries.groupby(['instrument', 'icb_industry_code', 'emission_intensity_col']).first().reset_index()
first_entries['year_lag'] = first_entries['year'] - 1
first_entries = first_entries[first_entries['year_lag'] > min(first_entries['year'])] # Drop any observation where no t-1 data could exist

# Emission intensity medians by industry-year
emission_medians_by_industry = data.groupby(['year', 'icb_industry_code'])[emissions_intensity_cols].median().reset_index()
emission_medians_by_industry = emission_medians_by_industry.melt(id_vars=['year', 'icb_industry_code'], var_name='emission_intensity_col', value_name='lagged_emission_intensity_sector_median').dropna().reset_index(drop=True)
emission_medians_by_industry.rename(columns={'year':'year_lag'}, inplace=True)

# Consolidate dataframes
industry_median_model = first_entries.merge(emission_medians_by_industry,
                                            how='left',
                                            left_on=['icb_industry_code', 'emission_intensity_col', 'year_lag'],
                                            right_on=['icb_industry_code', 'emission_intensity_col', 'year_lag'])
industry_median_model['absolute_error'] = abs(industry_median_model['value'] - industry_median_model['lagged_emission_intensity_sector_median'])

res = industry_median_model[industry_median_model['absolute_error'].isna()]
print(res)
