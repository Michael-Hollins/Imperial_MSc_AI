# Use S3 total to explore some outliers, where the model struggles
import numpy as np
import pandas as pd
from ml_pipeline import *
from clean_raw_data import *
import itertools
import os
import seaborn as sns
import matplotlib.patches as patches
from load_raw_data import *
from matplotlib.ticker import LogLocator, FuncFormatter

# Preliminaries
sns.set_style("whitegrid")
data_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/data'
figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/report/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300
refresh_plots = True

# Data preparation
df = pd.read_csv('src/results/predictions/s3_cat_total_intensity_core_material_2016_2022_log.csv')
df = df[['instrument', 's3_cat_total_intensity', 'XGB_PE', 'XGB']]
df['XGB_abs_error'] = df['XGB_PE'].abs()
df.sort_values(by='XGB_abs_error', inplace=True, ascending=False)
df.reset_index(inplace=True)
log_true_values = np.log(df['s3_cat_total_intensity'])
log_XGB_predictions = np.log(df['XGB'])

firm_A = df.loc[0, 'instrument']
firm_B = df.loc[6, 'instrument']
firm_C = df.loc[54, 'instrument']
outliers = [firm_A, firm_B, firm_C]

firm_A_estimates = df[df['instrument'] == firm_A]
firm_B_estimates = df[df['instrument'] == firm_B]
firm_C_estimates = df[df['instrument'] == firm_C]

if refresh_plots:
    fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*1.5), dpi=dpi)
    plt.scatter(log_XGB_predictions, log_true_values, alpha=0.5, label='', s=5)
    plt.scatter(np.log(firm_A_estimates['XGB']), np.log(firm_A_estimates['s3_cat_total_intensity']), color='orange', alpha=0.8, label='Firm A', edgecolor='black', s=13)
    plt.scatter(np.log(firm_B_estimates['XGB']), np.log(firm_B_estimates['s3_cat_total_intensity']), color='purple', alpha=0.8, label='Firm B', edgecolor='black', s=13)
    plt.scatter(np.log(firm_C_estimates['XGB']), np.log(firm_C_estimates['s3_cat_total_intensity']), color='green' , alpha=0.8, label='Firm C', edgecolor='black', s=13)

    # Add the 45-degree dashed line
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min(), log_true_values.max()], 
            'r--', linewidth=0.75, label='')

    # plus or minus 100%
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min() + np.log(2), log_true_values.max() + np.log(2)], 
            'gray', linestyle='--', linewidth=0.25, label='+100% Error')
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min() - np.log(2), log_true_values.max() - np.log(2)], 
            'gray', linestyle='--', linewidth=0.25, label='-100% Error')


    plt.xlabel('Log S3 cat total intensity predicted', fontsize=9)
    plt.ylabel('Log S3 cat total intensity actual', fontsize=9)
    plt.legend(loc='upper left',  frameon=False, ncol=1, title="", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_cat_total_outliers.png'))
    plt.close()

print(f'Firm A: {firm_A}')
print(f'Firm B: {firm_B}')
print(f'Firm C: {firm_C}')


# Investigating Firm A
data = pd.read_csv('data/ftse_world_allcap_clean.csv')
data = data[get_material_s3_category_reporting(data)]
scope_aggregates = data.loc[data['instrument'] == firm_A, ['year', 's1_co2e', 's2_co2e', 's3_cat_total']].dropna()
scope_aggregates = scope_aggregates.melt(id_vars='year', var_name='scope')

if refresh_plots:
    fig, ax = plt.subplots(figsize=(1.5*plot_width_inch, 1.5*plot_height_inch), dpi=dpi)
    sns.lineplot(data=scope_aggregates, x='year', y='value', hue='scope', marker='o')

    # Set plot labels and title
    plt.xlabel('')
    plt.ylabel('CO2e (tonnes)')
    plt.title('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc='upper right',  frameon=False, ncol=1, title="", fontsize=7)
    plt.ylabel('CO2e (tonnes)', fontsize=7)
    plt.tight_layout()
    plt.show()

s3_cat_cols = get_s3_cat_cols(data)
firm_A_cat_data = data.loc[data['instrument'] == firm_A, ['year'] + s3_cat_cols]
firm_A_cat_data = firm_A_cat_data.melt(id_vars='year', var_name='s3_category').dropna()
firm_A_cat_data = firm_A_cat_data.pivot_table(values='value', index='year', columns='s3_category', aggfunc='sum')

if refresh_plots:
    fig, ax = plt.subplots(figsize=(1.5*plot_width_inch, 1.5*plot_height_inch), dpi=dpi)
    firm_A_cat_data.plot(kind='bar', stacked=True, ax=ax, color=plt.cm.tab20.colors, width=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Stacked Bar Chart by S3 Category')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.xlabel('')
    plt.ylabel('CO2e (tonnes)')
    plt.title('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc='upper left',  frameon=False, ncol=1, title="", fontsize=5)
    plt.ylabel('CO2e (tonnes)', fontsize=7)
    plt.tight_layout()
    plt.show()

sub = data.loc[data['business_sector'] == 'Industrial Goods', ['year', 's3_cat_total_intensity', 'instrument', 'mcap']].dropna()
sub['log_mcap'] = np.log(sub['mcap'])
sub['log_s3_cat_total_intensity'] = np.log(sub['s3_cat_total_intensity'])
df_firm_A = sub[sub['instrument'] == firm_A]
df_ex_firm_A = sub[sub['instrument'] != firm_A]
if refresh_plots:
    fig, ax = plt.subplots(figsize=(1.5*plot_width_inch, 1.5*plot_height_inch), dpi=dpi)
    sns.scatterplot(x=df_ex_firm_A['log_mcap'], y=df_ex_firm_A['log_s3_cat_total_intensity'], s=10, alpha=0.5, label=None)
    sns.scatterplot(x=df_firm_A['log_mcap'], y=df_firm_A['log_s3_cat_total_intensity'], s=10, color='orange', alpha=1, label=f'Firm A')
    plt.title('')
    plt.xlabel('Log market cap', fontsize=10)
    plt.ylabel('Log S3 cat total CO2e intensity', fontsize=10)
    plt.legend(title='', fontsize=10, loc='lower right', frameon=True, ncol=1)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_cat_total_outliers_firm_A_mcap.png'))
    plt.close()
    
    
# Investigating Firm B
data = pd.read_csv('data/ftse_world_allcap_clean.csv')
if refresh_plots:
    s3_cat_cols_subset = ['s3_purchased_goods_cat1_intensity', 's3_capital_goods_cat2_intensity', 's3_waste_cat5_intensity', 's3_use_of_sold_cat11_intensity']
    firm_B_cat_data = data.loc[data['instrument'] == firm_B, ['year'] + s3_cat_cols_subset]
    firm_B_cat_data = firm_B_cat_data.melt(id_vars='year', var_name='s3_category').dropna()
    firm_B_cat_data = firm_B_cat_data.pivot_table(values='value', index='year', columns='s3_category', aggfunc='sum')
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    firm_B_cat_data.plot(kind='bar', stacked=True, ax=ax, color=plt.cm.tab20.colors, width=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    for patch in ax.patches:
        patch.set_edgecolor('black')  # Set the edge color to black
        patch.set_linewidth(1) 
    plt.xlabel('')
    plt.ylabel('S3 Intensity')
    plt.title('')
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right',  frameon=False, ncol=1, title="", fontsize=5)
    plt.ylabel('S3 Intensity (CO2e/USD mil)', fontsize=10)
    plt.tight_layout()
    plt.close()


if refresh_plots:
    firm_B_raw = pd.read_csv('data/firm_B_raw.csv')
    main_categories = [
        's3_purchased_goods_cat1_intensity',  # Category 1
        's3_capital_goods_cat2_intensity',    # Category 2
        's3_waste_cat5_intensity',            # Category 5
        's3_use_of_sold_cat11_intensity'      # Category 11
    ]

    other_categories = [
        's3_fuel_energy_cat3_intensity', 's3_transportation_cat4_intensity',
        's3_business_travel_cat6_intensity', 's3_employee_commuting_cat7_intensity',
        's3_leased_assets_cat8_intensity', 's3_distribution_cat9_intensity',
        's3_processing_products_cat10_intensity', 's3_EOL_treatment_cat12_intensity',
        's3_leased_assets_cat13_intensity', 's3_franchises_cat14_intensity',
        's3_investments_cat15_intensity'
    ]

    firm_B_raw['Other'] = firm_B_raw[other_categories].sum(axis=1)
    firm_B_grouped = firm_B_raw[['year'] + main_categories + ['Other']]
    firm_B_grouped.set_index('year', inplace=True)

    # Shades of blue for categories 1, 2, and 5, green for category 11, and grey for "Other"
    colors = ['#1f77b4', '#aec7e8', '#6baed6', '#2ca02c', '#7f7f7f']  # Customize as needed
    legend_labels = [
        'Purchased Goods (Cat 1)',   # For 's3_purchased_goods_cat1_intensity'
        'Capital Goods (Cat 2)',     # For 's3_capital_goods_cat2_intensity'
        'Waste (Cat 5)',             # For 's3_waste_cat5_intensity'
        'Use of Sold Products (Cat 11)',  # For 's3_use_of_sold_cat11_intensity'
        'Other'                      # For 'Other' category (grouped smaller categories)
    ]
    fig, ax = plt.subplots(figsize=(plot_width_inch*1.5, plot_height_inch*1.5))
    firm_B_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=np.e))

    # Set formatter to display natural log ticks as base e exponents
    def natural_log_formatter(y, pos):
        return f'{np.log(y):.0f}'  # Display as x
    ax.yaxis.set_major_formatter(FuncFormatter(natural_log_formatter))
    ax.set_xlabel('')
    ax.set_ylabel('Log Intensity (CO2e)', fontsize=10)
    ax.set_title('')
    ax.legend(loc='upper left', labels=legend_labels, frameon=False)
    for patch in ax.patches:
        patch.set_edgecolor('black') 
        patch.set_linewidth(1)
    plt.xticks(rotation=0)  
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_cat_total_outliers_firm_B_raw_data.png'))
    plt.close()

# Firm C

data = pd.read_csv('data/ftse_world_allcap_clean.csv')
data = data.loc[data['business_sector'] == 'Cyclical Consumer Products', ['year', 's3_cat_total_intensity', 'instrument']].dropna()
data['log_s3_cat_total_intensity'] = np.log(data['s3_cat_total_intensity'])
firm_C = data[data['instrument'] == firm_C]
others = data[~(data['instrument'].isin([firm_C]))]

if refresh_plots:
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    sns.scatterplot(x='year', y='log_s3_cat_total_intensity', data=others, alpha=0.6)
    plt.scatter(firm_C['year'], firm_C['log_s3_cat_total_intensity'], color='green', edgecolor='black', label='Firm C', zorder=5)
    plt.xlabel('')
    plt.ylabel('Log S3 Intensity', fontsize=7)
    plt.title('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, None)
    ax.grid(None)
    plt.legend(loc='upper left',  frameon=True, ncol=1, title="", fontsize=5)
    plt.tight_layout()    
    fig.savefig(os.path.join(figure_path, 's3_cat_total_outliers_firm_C_intensity.png'))
    plt.close()