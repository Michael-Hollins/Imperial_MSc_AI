import pandas as pd
import numpy as np

def audit_value_types(dt, axis_val=0, verbose=True):
    if axis_val not in [0, 1]:
        return "axis_val must be either 0 (cols) or 1 (rows)"
    
    if axis_val == 0:
        divisor = len(dt)
        index_val = dt.columns
        zeros = dt.isin([0]).sum(axis=axis_val) / divisor
        nulls = dt.isnull().sum(axis=axis_val) / divisor
        all_else = np.logical_and(dt.notnull(), dt != 0).sum(axis=axis_val) / divisor
        summary = pd.DataFrame(data={'zeros': zeros, 'nulls': nulls, 'all_else': all_else}, index=index_val)
    
    elif axis_val == 1:
        group = dt.groupby(['date_val', 'instrument'])
        zeros = group.apply(lambda x: (x == 0).sum().sum() / (len(x.columns) * len(x)))
        nulls = group.apply(lambda x: x.isnull().sum().sum() / (len(x.columns) * len(x)))
        all_else = group.apply(lambda x: np.logical_and(x.notnull(), x != 0).sum().sum() / (len(x.columns) * len(x)))
        summary = pd.DataFrame({'zeros': zeros, 'nulls': nulls, 'all_else': all_else})
    if verbose:
        print("")
        print('===================')
        print('VALUE AUDIT')
        print('===================')
        print("")
        print(summary)


def compare_s3_categories_with_total(df, verbose = True):
    s3_category_cols = df.loc[:, 's3_purchased_goods_cat1':'s3_investments_cat15'].columns
    df['sum_across_s3_categories'] = df[s3_category_cols].sum(axis=1)
    # If a firm reports zero S3 emissions across their categories, we change the value to NA
    df.loc[data['sum_across_s3_categories'] == 0, 'sum_across_s3_categories'] = np.nan 
    both_missing = np.logical_and(np.isnan(df['sum_across_s3_categories']), np.isnan(df['s3_co2e']))
    one_missing = np.isnan(df['sum_across_s3_categories']) != np.isnan(df['s3_co2e'])
    neither_missing = np.logical_and(~np.isnan(df['sum_across_s3_categories']), ~np.isnan(df['s3_co2e']))
    df['is_sum_equal_to_s3_co2e'] = df['sum_across_s3_categories'] == df['s3_co2e']
    non_matching_totals = df.loc[np.logical_and(df['is_sum_equal_to_s3_co2e'] == False, neither_missing), ['date_val', 'instrument', 'firm_name', 'sum_across_s3_categories', 's3_co2e']]
    non_matching_totals['discrepency_perc'] = 100*abs(non_matching_totals['sum_across_s3_categories'] - non_matching_totals['s3_co2e'])/non_matching_totals['s3_co2e']
    non_matching_totals.sort_values(by='discrepency_perc', ascending=False, inplace=True, ignore_index=True)
    if verbose:
        print("")
        print('===================')
        print('COMPARING TOTAL S3 FIELD TO SUM OF S3 CATGEGORICAL FIELDS')
        print('===================')
        print("")
        print(f"We begin with {len(df)} observations")
        print('')
        print(f"{sum(both_missing)} observations ({100*sum(both_missing)/len(df):.2f}%) have missing both the total S3 amount and the implied total over the categories")
        print('')
        print(f"{sum(one_missing)} observations ({100*sum(one_missing)/len(df):.2f}%) have either the total S3 amount missing or the totalled up amount missing")
        print('')
        print(f"Therefore we can only compare the numbers for {sum(neither_missing)} ({100*sum(neither_missing)/len(df):.2f}%) cases")
        print('')
        print(f"Of these, {sum(df['is_sum_equal_to_s3_co2e'])} observations ({100*sum(df['is_sum_equal_to_s3_co2e'])/len(df):.2f}%) have a matching total field and implied category total.")
        print('')
        print(f"The mean discrepency is {np.mean(non_matching_totals['discrepency_perc']):.2f}% and the median is {np.median(non_matching_totals['discrepency_perc']):.2f}%")
        print('')
        print("The top ten largest discrepencies are...")
        print('')
        print(non_matching_totals.head(10))
        
def get_firms_per_year():
    pass

def get_new_firms_per_year():
    pass       
    
if __name__=="__main__":
    data = pd.read_pickle('data/mock_universe/mock_universe.pkl')
    audit_value_types(data, axis_val=0)
    compare_s3_categories_with_total(data)