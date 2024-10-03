import pandas as pd
from clean_raw_data import get_emissions_intensity_cols
import re

def get_median_intensities_per_sector(df, group_col, verbose=True):
    """
    Calculate the median intensities of Scope 3 emissions per sector.

    This function groups the DataFrame by the specified sector field and calculates
    the median intensity for each Scope 3 emissions category within each sector.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing emissions intensity data.
                       Expected columns include various 's3_' columns ending with '_intensity'.
    group_col (str): The column name to group by, representing different sectors.

    Returns:
    pd.DataFrame: The DataFrame with the median intensities of Scope 3 emissions for each sector.
    """
    all_intensities = get_emissions_intensity_cols(df)
    s3_intensities = [x for x in all_intensities if re.search(r'_cat\d+', x)]
    result = df.groupby([group_col])[s3_intensities].median()
    if verbose:
        print(result)
    return result


def get_intensity_rank_per_sector(df, verbose=True):
    """
    Rank Scope 3 emissions intensities per sector.

    This function ranks the intensity of Scope 3 emissions categories within each sector,
    assigning ranks in descending order (highest intensity gets rank 1) while preserving the original intensity values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing median emissions intensity data by sector.

    Returns:
    pd.DataFrame: The DataFrame with both the original intensities and the ranks for each Scope 3 emissions category within each sector.
    """
    rank_df = df.T.rank(ascending=False).T
    rank_df = rank_df.add_suffix('_rank')
    combined_df = pd.concat([df, rank_df], axis=1)
    combined_df = combined_df.reset_index()
    if verbose:
        print(combined_df)
    return combined_df


def return_s3_cat_importance_per_sector(df, group_col, verbose=True):
    """
    Determine the importance of Scope 3 categories per sector.

    This function melts the DataFrame, extracts the category number from the category name,
    sorts the categories by rank within each sector, and groups them to list the importance
    of each category along with their corresponding intensity values.

    Parameters:
    df (pd.DataFrame): The input DataFrame with ranked Scope 3 emissions intensities.
    group_col (str): The column name to group by, representing different sectors.

    Returns:
    pd.DataFrame: The DataFrame with the ordered list of category importance per sector,
                  their corresponding intensity values, and the top two categories highlighted.
    """
    def extract_cat_number(col_value):
        match = re.search(r'cat(\d+)_intensity', col_value)
        if match:
            return match.group(1)
        return None

    # Ensure correct columns are used for melting and ranking
    value_cols = [col for col in df.columns if 'intensity' in col and '_rank' not in col]
    rank_cols = [col for col in df.columns if '_rank' in col]

    # Continue with melting and processing
    rank_df = df.melt(id_vars=[group_col], value_vars=rank_cols, var_name='category', value_name='rank')
    rank_df['category'] = rank_df['category'].apply(extract_cat_number)
    value_df = df.melt(id_vars=[group_col], value_vars=value_cols, var_name='category', value_name='median_value')
    value_df['category'] = value_df['category'].apply(extract_cat_number)

    # Merge rank and value dataframes
    df = pd.merge(rank_df, value_df, on=[group_col, 'category'])
    df = df.groupby(group_col).apply(lambda x: x.sort_values('rank')).reset_index(drop=True)
    
    grouped_df = df.groupby(group_col).apply(
        lambda x: pd.Series({
            's3_category_importance': list(x['category']),
            's3_category_median_values': list(round(x['median_value'],2))
        })
    ).reset_index()
    grouped_df.columns = ['sec_code', 's3_category_importance', 's3_category_values']
    grouped_df['top_two_categories'] = grouped_df['s3_category_importance'].apply(lambda x: x[:2])
    if verbose:
        print(grouped_df)
    return grouped_df


def get_s3_cat_rank_per_sector(df, group_cols=['econ_sector_code', 'business_sector_code']):
    """
    Get the ranking of Scope 3 categories per sector.

    This function processes the DataFrame to calculate the Scope 3 emissions intensities,
    determines the median intensities per sector, ranks these intensities, and returns
    the importance of each category per sector along with their corresponding intensity values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing emissions data.
    group_cols (list): A list of column names to group by, representing different sectors.

    Returns:
    pd.DataFrame: The DataFrame with the ordered list of category importance per sector,
                  their corresponding intensity values, and the top two categories highlighted,
                  for each sector specified in group_cols.
    """
    s3_cat_ranks = list()
    for grp in group_cols:
        s3_median_intensities = get_median_intensities_per_sector(df, grp, verbose=False)
        s3_median_intensities_ranked = get_intensity_rank_per_sector(s3_median_intensities, verbose=False)
        s3_median_intensities_ranked_tidy = return_s3_cat_importance_per_sector(s3_median_intensities_ranked, grp, verbose=False)
        s3_cat_ranks.append(s3_median_intensities_ranked_tidy)
    res = pd.concat(s3_cat_ranks, ignore_index=True)
    res['sec_code'] = res['sec_code'].astype(int).astype(str)
    res = res.sort_values(by='sec_code').reset_index(drop=True)
    return res


def s3_cat_intensity_proportion_by_sector_materiality(df, observations, sector):
    # Initialise a list to store results
    res = list()
    
    # Define some column names
    observation_cols = observations
    sector_col = sector
    s3_cat_intensity_cols = get_emissions_intensity_cols(df)
    
    # Get the total carbon intensity and work out proportions
    df['total_carbon_intensity'] = df[s3_cat_intensity_cols].sum(axis=1)
    df = df[df['total_carbon_intensity'] > 0] # Remove where we have no S3 category data
    for col in s3_cat_intensity_cols:
        df.loc[f'{col}_proportion'] = df[col] / df['total_carbon_intensity']
    cat_intensity_prop_cols = get_emissions_intensity_cols(df)
    df = df[df[sector_col].notnull()] # remove where sectors are missing
    
    # Get the S3 category importance by sector 
    s3_cat_ranks = get_s3_cat_rank_per_sector(df, group_cols=[sector_col])
    
    # For each sector, map the ranks to materiality 
    sectors = df[sector_col].unique()
    for sec in sectors:
        sector_df = df.loc[df[sector_col] == sec, observation_cols + [sector_col] + cat_intensity_prop_cols]
        
        pattern = r'(cat\d+)'
        new_col_names = list(sector_df.columns.str.extract(pattern, expand=False))[3:]
        col_name_mapping = {k:v for k, v in zip(get_emissions_intensity_cols(sector_df), new_col_names)}
        sector_df.rename(columns=col_name_mapping, inplace=True)
        
        cat_importance = s3_cat_ranks.loc[s3_cat_ranks['sec_code'] == str(int(sec)), 's3_category_importance']
        cat_ordering_by_importance = ['cat' + str(x) for y in cat_importance for x in y]
        
        sector_df = sector_df.loc[:, observation_cols + [sector_col] + cat_ordering_by_importance]
        most_material_col_names = ['most_material_' + str(x) for x in range(1, 16)]
        sector_df.columns = observation_cols + [sector_col] + most_material_col_names
        
        res.append(sector_df)
    
    result = pd.concat(res, ignore_index=True)
    return result

if __name__=="__main__":
    # The aim here is to replicate the materiality analysis in this paper:
    # https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/solving-scope-3-conundrum.pdf
    
    # data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    # res = get_s3_cat_rank_per_sector(data, group_cols=['business_sector_code'])
    # cat_cols = ['s3_purchased_goods_cat1_intensity', 's3_capital_goods_cat2_intensity', 's3_fuel_energy_cat3_intensity',
    #             's3_transportation_cat4_intensity', 's3_waste_cat5_intensity', 's3_business_travel_cat6_intensity',
    #             's3_employee_commuting_cat7_intensity', 's3_leased_assets_cat8_intensity', 's3_distribution_cat9_intensity',
    #             's3_processing_products_cat10_intensity', 's3_use_of_sold_cat11_intensity', 's3_EOL_treatment_cat12_intensity',
    #             's3_leased_assets_cat13_intensity', 's3_franchises_cat14_intensity', 's3_investments_cat15_intensity']
    # for col in cat_cols:
    #     data.loc[f'{col}_proportion'] = data[col] / data['s3_cat_total_intensity']
        
        
        
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