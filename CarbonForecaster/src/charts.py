import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from eda import get_s3_prop_by_industry
from clean_raw_data import get_s3_cat_cols
from clean_raw_data import get_sector_cols
from clean_raw_data import get_material_s3_category_reporting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import iqr
from ml_pipeline import *
import pandas as pd

# Set parameters
sns.set_style("whitegrid")
data_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/data'
figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/report/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300

REFRESH_PLOTS = False

def plot_remove_labels(plot):
    plot.set_xlabel("")
    plot.set_ylabel("")
    plot.set_title("")

######################################################
if REFRESH_PLOTS:
    total_ghg_emissions = pd.read_csv(os.path.join(data_path, 'total-ghg-emissions.csv'))

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    world_ghg_emissions = sns.lineplot(data = total_ghg_emissions[total_ghg_emissions['Entity'] == 'World'],
                x = 'Year',
                y = 'Annual greenhouse gas emissions in CO₂ equivalents')
    plot_remove_labels(world_ghg_emissions)
    world_ghg_emissions.set_yticks(ticks = [0, 1e10, 2e10, 3e10, 4e10, 5e10, 6e10])
    ylabels = ['{:,.0f}'.format(y) for y in world_ghg_emissions.get_yticks()/1000000000]
    world_ghg_emissions.set_yticklabels(ylabels)
    world_ghg_emissions.figure.tight_layout()
    sns.despine(left=True)
    fig = world_ghg_emissions.get_figure()
    fig.savefig(os.path.join(figure_path, 'world_ghg_emissions.png'), dpi=dpi)
    plt.close()

######################################################
if REFRESH_PLOTS:
    temp_anomalies = pd.read_csv(os.path.join(data_path, 'temperature-anomaly.csv'))

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    world_temp_anomalies = sns.lineplot(data=temp_anomalies[temp_anomalies['Entity']=='Global'],
                                        x = 'Year',
                                        y = 'Global average temperature anomaly relative to 1961-1990')
    plot_remove_labels(world_temp_anomalies)
    world_temp_anomalies.figure.tight_layout()
    sns.despine(left=True)
    fig = world_temp_anomalies.get_figure()
    fig.savefig(os.path.join(figure_path, 'world_temp_anomalies.png'), dpi=dpi)
    plt.close()

#######################################################
if REFRESH_PLOTS:
    # Scope 3 coverage
    # Source: LSEG's "Solving the Scope 3 condundrum"

    years = [2016, 2017, 2018, 2019, 2020, 2021]
    scope_1_2 = [63, 68, 72, 65, 69, 70]
    scope_3 = [37, 37, 42, 38, 41, 45]
    material_scope_3 = [13, 14, 16, 15, 18, 20]

    data = {
        'Year': years,
        'Scope 1&2': scope_1_2,
        'Scope 3': scope_3,
        'Material Scope 3': material_scope_3
    }

    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Year', var_name='Scope', value_name='Percentage')

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    data_coverage = sns.lineplot(data=df_melted, x='Year', y='Percentage', hue='Scope', marker='o')
    plot_remove_labels(data_coverage)
    data_coverage.figure.tight_layout()
    plt.xticks(years)
    plt.yticks(range(0, 101, 20))
    plt.legend(title="", loc='upper left', fontsize='xx-small')
    fig = data_coverage.get_figure()
    fig.savefig(os.path.join(figure_path, 'data_coverage.png'), dpi=dpi)
    plt.close()

#############################################################
if REFRESH_PLOTS:
    # 2021 S3 data volatility
    # Source: Figure 9 https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/solving-scope-3-conundrum.pdf
    figures_2021 = [44, 21, 17, 18]
    categories = ['< 20%', '20% - 50%', '50% - 100%', '> 100%']
    colours = ['blue', 'lightblue', 'grey', 'darkblue']

    # Creating a horizontal bar plot with combined segments, annotations, and legend
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    position = 0
    bars = []
    for value, color, label in zip(figures_2021, colours, categories):
        bars.append(plt.barh(0, value, left=position, color=color, edgecolor='black', label=label))
        # Adding text annotations in the middle of each segment
        plt.text(position + value / 2, 0, f'{value}%', va='center', ha='center', color='white', fontsize=12, fontweight='bold')
        position += value

    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Set labels and title
    fig.tight_layout()
    plt.xlim(0, 100)
    plot_remove_labels(ax)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize='small')
    fig.savefig(os.path.join(figure_path, 's3_2021_vol.png'), dpi=dpi)
    plt.close()

#######################################################
if REFRESH_PLOTS:
    # Scope 3 share by upstream and downstream splits by sector
    data = pd.read_csv('data/ftse_350/clean_from_excel.csv')
    data = get_s3_prop_by_industry(data, verbose=False)
    data['downstream'] = data['upstream'] + data['downstream']
    data = data.sort_values('downstream', ascending=True)

    fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*2), dpi=dpi)
    sns.set_theme(font_scale=0.9)
    sns.set_color_codes("pastel")
    sns.barplot(x="downstream", y="icb_industry_name", data=data, label="Downstream", color="b", edgecolor='black')
    sns.set_color_codes("muted")
    sns.barplot(x="upstream", y="icb_industry_name", data=data, label="Upstream", color="b", edgecolor='black')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    handles, labels = ax.get_legend_handles_labels()
    ax.grid(False)
    ax.legend(handles[::-1], labels[::-1], ncol=1, loc="upper right", frameon=False, fontsize=9)
    ax.set(xlim=(0, 1), ylabel="", xlabel="")
    plt.tick_params(axis='both', which='major', labelsize=7)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    fig.savefig(os.path.join(figure_path, 's3_upstream_downstream_shares_by_sector.png'), dpi=dpi)
    plt.close()
  
#######################################################
if REFRESH_PLOTS:
    # Number of instruments (firms?) per year
    data = pd.read_csv('data/ftse_global_allcap_clean.csv')
    instruments_by_year = data.groupby(['year'])['instrument'].nunique().reset_index()
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    sns.set_color_codes("pastel")
    sns.lineplot(data=instruments_by_year , x='year', y='instrument')
    plot_remove_labels(ax)
    plt.axes()
    fig.savefig(os.path.join(figure_path, 'instruments_by_year.png'), dpi=dpi)
    plt.close()

#######################################################
if REFRESH_PLOTS:
    # Industry breakdown in 2022 - last year in our dataset
    data = pd.read_csv('data/ftse_global_allcap_clean.csv')
    industry_breakdown_2022 = data.loc[data['year'] == 2022, 'icb_industry_name'].value_counts().reset_index()
    industry_breakdown_2022.columns = ['industry', 'count']
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    sns.set_color_codes("pastel")
    sns.barplot(data=industry_breakdown_2022, x='count', y='industry')
    plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plot_remove_labels(ax)
    fig.savefig(os.path.join(figure_path, 'industry_breakdown_2022.png'), dpi=dpi)
    plt.close()

#######################################################
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    proportion_non_null_by_year = data.groupby('year')[['s1_co2e', 's2_co2e', 's3_co2e', 's3_upstream', 's3_downstream']].apply(lambda x: x.notnull().mean()).reset_index()
    proportion_non_null_by_year = proportion_non_null_by_year.melt(id_vars='year', var_name='scope', value_name='value')

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    sns.lineplot(data=proportion_non_null_by_year, x='year', y='value', hue='scope', marker='o')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(-0.1, 1)  
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title='', loc='upper left', fontsize='xx-small', frameon=False)
    plot_remove_labels(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 'scope_disclosure_by_year.png'), dpi=dpi)
    plt.close()
    
#######################################################
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    fundamentals = ['mcap', 'revenue', 'ebit', 'ebitda', 'net_cash_flow', 'assets', \
                    'receivables', 'net_ppe', 'capex', 'intangible_assets', 'lt_debt']
    emission_scores = ['policy_emissions_score', 'target_emissions_score']
    X = data[fundamentals + emission_scores].dropna()
    correlation_matrix = X.corr()

    # Plotting the correlation matrix
    fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*2), dpi=dpi)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature correlation matrix')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 'feature_correlation_matrix.png'), dpi=dpi)
    plt.close()
    
#######################################################
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data['s1_and_s2_co2e'] = np.log(data['s1_and_s2_co2e'])
    data['s3_cat_total'] = np.log(data['s3_cat_total'])
    filtered_data = data.dropna(subset=['s1_and_s2_co2e', 's3_cat_total'])

    # Prepare the data
    X_filtered = filtered_data['s1_and_s2_co2e'].values.reshape(-1, 1)
    y_filtered = filtered_data['s3_cat_total'].values

    # Fit the linear regression model
    model_filtered = LinearRegression()
    model_filtered.fit(X_filtered, y_filtered)

    # Predict the y values
    y_pred_filtered = model_filtered.predict(X_filtered)

    # Calculate the R-squared value
    r2_filtered = r2_score(y_filtered, y_pred_filtered)

    # Plot the data and the line of best fit
    fig, ax = plt.subplots(figsize=(plot_width_inch*1.75, plot_height_inch*1.75), dpi=dpi)
    sns.scatterplot(x=filtered_data['s1_and_s2_co2e'], y=filtered_data['s3_cat_total'], s=10, label=None)
    plt.plot(filtered_data['s1_and_s2_co2e'], y_pred_filtered, color='red', label=f'Line of Best Fit (R² = {r2_filtered:.2f})')
    plt.title('')
    plt.xlabel('Log of S1 + S2 CO2e', fontsize=7)
    plt.ylabel('Log of S3 Category Total CO2e', fontsize=7)
    plt.xlim(4, 22)
    plt.ylim(4, 22)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(title='', fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 'scope_regression.png'), dpi=dpi)
    plt.close()

#######################################################
# Sankey diagram for sample construction
#######################################################
if True:
    labels=['Load from API', 'No fundamentals', '', # 0, 1, 2
        'Dual-listed duplicates', '', # 3, 4
        'S1 or S2 reported values missing', '', # 5, 6
        'Removing financial firms', '', # 7, 8
        'Sectors with too few peers', '', # 9, 10
        'Non-normal obs', # 11
        'Remaining sample'] # 12
    colours = ['green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green']
    sources = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    values = [3264, 50712, 2209, 48503, 34691, 13812, 2205, 11607, 227, 11330, 532, 10798]

    x_positions = [-0.1,  0.15, 0.15, 0.3, 0.3,  0.45, 0.45, 0.6, 0.6, 0.7, 0.7, 1, 1]
    y_positions = [0.55, 0.1,  0.55, 0.75, 0.55, 0.15, 0.55, 0.65, 0.55, 0.3, 0.55, 0.4, 0.55]
    fig = go.Figure(data=[go.Sankey(
        arrangement = "snap",
        node = dict(
            pad = 15, 
            thickness = 20,
            line = dict(color = 'black', width=0.5),
            label = labels,
            color = colours,
            x = x_positions,
            y = y_positions
        ),
        link = dict(
            source = sources,
            target = targets,
            value = values
        )
    )])
    
    # Update the layout to minimize margins
    fig.update_layout(
        autosize=False,  # Disable autosizing to control dimensions manually
        width=10,    # Set the desired width
        height=10,  # Set the desired height
        margin=dict(l=100, r=20, t=20, b=20)  # Reduce margins (adjust as needed)
    )

    pio.write_image(fig = fig, file = os.path.join(figure_path, 'preprocessing_sankey.png'), format='png', width = plot_width_inch*1.5, height = plot_height_inch*1.5)

#######################################################
if REFRESH_PLOTS:    
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    sub = data[(data['year'] == 2022) & (data['s3_cat_total'].notnull())]
    sub_cat3 = sub.groupby(['econ_sector'])['s3_cat_total'].sum().reset_index().sort_values(by='s3_cat_total', ascending=False)
    sub_num_share = sub['econ_sector'].value_counts(normalize=True).reset_index()
    sub_num_share.columns = ['econ_sector', 'index_num_share']
    hist_df = sub_cat3.merge(sub_num_share, how='left', on='econ_sector')
    hist_df['x_pos'] = hist_df['index_num_share'].cumsum() - hist_df['index_num_share']

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    colors = cm.tab20(np.linspace(0, 1, len(hist_df)))
    bars = plt.bar(hist_df['x_pos'], hist_df['s3_cat_total'] / 1e9, width=hist_df['index_num_share'], color=colors, align='edge')

    # Adding labels
    plt.xlabel('Firm share', fontsize=7)
    plt.ylabel('Total S3 Emissions (bn tonnes)', fontsize=7)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.2))

    # Remove gridlines
    plt.grid(False)

    # Remove the gray border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # Adding a legend with two columns
    plt.legend(bars, hist_df['econ_sector'], loc='upper right', bbox_to_anchor=(1.15, 1), frameon=False, ncol=1, title="", fontsize=5)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 'sector_histogram.png'), dpi=dpi)
    plt.close()
    

#######################################################
if REFRESH_PLOTS:    
    s3_results = pd.read_csv('src/results/s3_upstreampredictions_and_APE.csv')
    s3_results = s3_results[['instrument', 's3_upstream', 'RF_y_pred_unscaled']]
    all_data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    sub = all_data[['instrument'] + get_sector_cols() + ['cc_classification']]
    s3_results = s3_results.merge(sub, how='left', on='instrument')

    s3_results['log_s3_upstream'] = np.log(s3_results['s3_upstream'])
    s3_results['log_RF_y_pred_unscaled'] = np.log(s3_results['RF_y_pred_unscaled'])

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='log_RF_y_pred_unscaled', y='log_s3_upstream', data=s3_results, hue='econ_sector',)

    # Add the 45-degree dashed line (perfect predictions)
    plt.plot([s3_results['log_s3_upstream'].min(), s3_results['log_s3_upstream'].max()], 
            [s3_results['log_s3_upstream'].min(), s3_results['log_s3_upstream'].max()], 
            'r--', linewidth=2, label='45-degree line')

    # Add ±100% error lines (grey dashed lines)
    plt.plot([s3_results['log_s3_upstream'].min(), s3_results['log_s3_upstream'].max()], 
            [s3_results['log_s3_upstream'].min() + np.log(2), s3_results['log_s3_upstream'].max() + np.log(2)], 
            'gray', linestyle='--', linewidth=1, label='+100% Error')
    plt.plot([s3_results['log_s3_upstream'].min(), s3_results['log_s3_upstream'].max()], 
            [s3_results['log_s3_upstream'].min() - np.log(2), s3_results['log_s3_upstream'].max() - np.log(2)], 
            'gray', linestyle='--', linewidth=1, label='-100% Error')

    plt.xlabel('Log RF_y_pred_unscaled')
    plt.ylabel('Log S3 Upstream')
    plt.title('Scatter Plot of Log RF_y_pred_unscaled vs Log S3 Upstream')
    plt.legend(title='CC Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#######################################################
# 2022 industry breakdown to illustrate sector imbalance
#######################################################
if REFRESH_PLOTS:     
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data = data[(data['year'] == 2022) & (data['s3_cat_total_intensity'].notnull())]
    sector_count = data.groupby('econ_sector').size().reset_index()
    sector_count.columns = ['econ_sector', 'count']
    sector_count.sort_values(by='count', inplace=True)

    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    sns.set_color_codes("pastel")
    sns.barplot(data=sector_count, x='count', y='econ_sector', ax=ax)

    # Remove grid lines and borders
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add the raw count numbers at the end of each bar
    for i in ax.patches:
        ax.text(i.get_width() + 0.3, i.get_y() + i.get_height() / 2, 
                f'{int(i.get_width())}', fontsize=7, color='black', va='center')

    # Remove the x-axis
    ax.xaxis.set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plot_remove_labels(ax)
    fig.savefig(os.path.join(figure_path, 'industry_breakdown_2022_s3_cat_total.png'), dpi=dpi)
    plt.close()

#######################################################
# Carbon intensity at econ sector level vs industry
#######################################################
if REFRESH_PLOTS:        
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data = data[(data['year'] == 2022) & (data['s3_cat_total_intensity'].notnull())]
    econ_sector_intensity_median = data.groupby('econ_sector')['s3_cat_total_intensity'].median().reset_index()
    econ_sector_intensity_median.columns = ['econ_sector', 'econ_sector_median']
    industry_sector_intensity_median = data.groupby(['econ_sector', 'industry_sector'])['s3_cat_total_intensity'].median().reset_index()
    industry_sector_intensity_median.columns = ['econ_sector', 'industry_sector', 'industry_sector_median']
    median_intensity = industry_sector_intensity_median.merge(econ_sector_intensity_median, how='left', on='econ_sector')
    median_intensity = median_intensity.sort_values(by='econ_sector_median', ascending=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*1.5), dpi=dpi)

    # Plot the smaller gray dots for industry sector medians
    for i, row in median_intensity.iterrows():
        ax.scatter(row['industry_sector_median'], row['econ_sector'], color='gray', s=10, alpha=0.6, label='Industry sector median' if i == 0 else "")

    # Plot the larger blue dots for econ sector medians
    econ_medians = median_intensity.drop_duplicates(subset=['econ_sector'])
    for i, row in econ_medians.iterrows():
        ax.scatter(row['econ_sector_median'], row['econ_sector'], color='blue', s=40, edgecolor='black', label='Economic sector median' if i == 0 else "")

    ax.set_xscale('log') # Set x-axis to log scale
    ax.set_xlabel('Carbon Intensity, Tonnes CO2e per Million USD, log10 scale')
    ax.set_ylabel('')
    ax.set_title('Carbon Intensity by Econ Sector and Industry Sector')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=False, ncol=2, fontsize=5)

    # Remove the gridlines and spines for a clean look
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel('Carbon Intensity, Tonnes CO2e per Million USD, log10 scale', fontsize=7)
    plt.title('')
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    fig.savefig(os.path.join(figure_path, 'econ_versus_industry_median_s3_intensity.png'), dpi=dpi)
    plt.close()


#######################################################
# Tradeoff between industry level coverage and volatility
#######################################################
if REFRESH_PLOTS:        
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data = data[(data['year'] == 2022) & (data['s3_cat_total_intensity'].notnull())]

    sector_columns = ['econ_sector', 'business_sector', 'industry_group_sector', 'industry_sector', 'activity_sector']

    # Prepare a list to hold results
    results = []

    # Loop over each sector column
    for sector in sector_columns:
        # Group by the sector and calculate the IQR, mean, and number of firms (size)
        sector_stats = data.groupby(sector).agg(
            iqr_s3_cat_total_intensity=('s3_cat_total_intensity', lambda x: iqr(x)),
            mean_s3_cat_total_intensity=('s3_cat_total_intensity', 'mean'),
            num_firms=('s3_cat_total_intensity', 'size')
        ).reset_index()
        
        # Normalize the IQR by dividing by the mean
        sector_stats['normalized_iqr'] = sector_stats['iqr_s3_cat_total_intensity'] / sector_stats['mean_s3_cat_total_intensity']
        
        # Calculate the median of the normalized IQR and the median firm count
        median_normalized_iqr = sector_stats['normalized_iqr'].median()
        median_firm_count = sector_stats['num_firms'].median()
        
        # Store the result
        results.append({
            'sector_level': sector,
            'median_normalized_iqr': median_normalized_iqr,
            'median_firm_count': median_firm_count
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)


    # Define a color palette for the different sector levels
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    # Create a dictionary mapping sector levels to their labels and colors
    sector_labels = {
        'econ_sector': ('Economic - Level 1', colors[0]),
        'business_sector': ('Business - Level 2', colors[1]),
        'industry_group_sector': ('Industry Group - Level 3', colors[2]),
        'industry_sector': ('Industry - Level 4', colors[3]),
        'activity_sector': ('Activity - Level 5', colors[4]) 
    }

    # Plotting the results
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)

    # Track which labels have been added
    labels_added = set()

    # Plot each sector group with its own color
    for i, row in results_df.iterrows():
        sector_label, color = sector_labels[row['sector_level']]
        if sector_label not in labels_added:
            ax.scatter(row['median_firm_count'], row['median_normalized_iqr'], color=color, s=10, label=sector_label)
            labels_added.add(sector_label)
        else:
            ax.scatter(row['median_firm_count'], row['median_normalized_iqr'], color=color, s=10)

    # Set axis labels and title
    ax.set_title('')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel('Median firm count per group', fontsize=7)
    plt.ylabel('Normalised median of IQR\nof S3 carbon intensity', fontsize=7)
    plt.ylim(0, 1.2)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  
    ax.legend(unique_labels.values(), unique_labels.keys(), title="", fontsize=5, title_fontsize=6,
            loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False, fancybox=False, shadow=False, ncol=2, markerscale=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Adjust the bottom to make space for the legend
    fig.savefig(os.path.join(figure_path, 'vol_vs_coverage_sector_level.png'), dpi=dpi)
    plt.close()
    
#######################################################
# Counts of GHG emissions 
######################################################
if REFRESH_PLOTS:
    data_file_path = 'data/ftse_world_allcap_clean.csv'
    data = pd.read_csv(data_file_path)
    data['material_s3'] = get_material_s3_category_reporting(data)
    s1_and_s2_count = data.groupby('year')['s1_and_s2_co2e'].apply(lambda x: x.notnull().sum())
    s3_cat_total_count = data.groupby('year')['s3_cat_total'].apply(lambda x: x.notnull().sum())
    material_s3_count = data.groupby('year')['material_s3'].apply(lambda x: (x == 1).sum())

    # Concatenate into a DataFrame
    custom_labels = ['S1 and S2', 'S3 cat', 'Material S3 cat']
    result_df = pd.concat([s1_and_s2_count, s3_cat_total_count, material_s3_count], axis=1)
    result_df.columns = ['s1_and_s2_co2e_count', 's3_cat_total_count', 'material_s3_count']
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    result_df.plot(kind='line', ax=ax)
    ax.legend(title='', loc='best', fontsize=7, labels=custom_labels, frameon=False)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 'ghg_observations_by_year.png'), dpi=dpi)
    plt.close()

#######################################################
# S3 category disclosure by mcap 
######################################################
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data = data[data['year'] == 2022]
    data['disclosed'] = data['s3_cat_total'].notnull()
    data = data.sort_values(by='mcap', ascending=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    colors = data['disclosed'].map({True: 'blue', False: 'grey'})
    ax.barh(range(len(data)), data['mcap'], color=colors, edgecolor='none', height=3)
    ax.set_xscale('log')
    ax.yaxis.set_visible(False)
    handles = [plt.Line2D([0], [0], color='blue', lw=4),
            plt.Line2D([0], [0], color='grey', lw=4)]
    labels = ['Disclosure', 'Missing']
    plt.legend(handles, labels, title='', fontsize=7, loc='best', frameon=False)
    plt.xlabel('Market Capitalisation (USD, log scale)', fontsize=7)
    plt.xticks(fontsize=7)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_disclosure_by_mcap.png'), dpi=dpi)
    plt.close()

#######################################################
# S3 category disclosure by sector incl materiality 
######################################################
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    data = data[data['year'] == 2022]
    data['disclosed'] = data['s3_cat_total'].notnull()
    data['material'] = get_material_s3_category_reporting(data)
    df = data.groupby('econ_sector').agg(n=('instrument', 'size'), disclosed=('disclosed', 'sum'), material=('material', 'sum')).reset_index()
    df['percent_disclosed'] = df['disclosed'] / df['n'] * 100
    df['percent_material'] = df['material'] / df['n'] * 100
    df = df.sort_values(by='percent_disclosed', ascending=False)
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    bar_width = 0.6
    indices = np.arange(len(df))

    # Plot bars
    ax.barh(indices, df['percent_disclosed'], color='lightblue', label='Reported', height=bar_width)
    ax.barh(indices, df['percent_material'], color='blue', label='Material', height=bar_width)
    ax.set_yticks(indices)
    ax.set_yticklabels(df['econ_sector'], fontsize=7)
    ax.set_xlabel('% of Firms', fontsize=7)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.invert_yaxis()
    ax.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.savefig(os.path.join(figure_path, 's3_disclosure_by_sector_incl_materiality.png'), dpi=dpi)
    plt.close()


#######################################################
# S3 category disclosure bars and dots 
######################################################    
if REFRESH_PLOTS:
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    total_s3_amount = data['s3_cat_total'].sum()
    s3_cat_data = data.loc[data['s3_cat_total'].notnull(), ['instrument'] + get_s3_cat_cols(data)]
    disclosure_rate = pd.DataFrame(s3_cat_data[get_s3_cat_cols(data)].notnull().mean()*100).reset_index()
    disclosure_rate.columns = ['category', 'disclosure_rate']
    disclosure_rate['category'] = disclosure_rate['category'].str.replace(r'.*_cat(\d+)', r'Cat \1', regex=True)
    disclosure_rate.sort_values(by='disclosure_rate', inplace=True, ascending=False)

    cat_perc = s3_cat_data.melt(id_vars='instrument', var_name='category', value_name='emissions_share')
    cat_perc['category'] = cat_perc['category'].str.replace(r'.*_cat(\d+)', r'Cat \1', regex=True)
    cat_perc.dropna(inplace=True)
    cat_perc = cat_perc.groupby(['category']).sum().reset_index()
    emissions_total = cat_perc['emissions_share'].sum()
    cat_perc['emissions_share'] = 100*cat_perc['emissions_share']/emissions_total

    df = disclosure_rate.merge(cat_perc, how='left', on='category')
    print(df)
    fig, ax = plt.subplots(figsize=(plot_width_inch, plot_height_inch), dpi=dpi)
    ax.barh(df['category'], df['emissions_share'], color='blue', alpha=0.8, label='Share in S3 emissions disclosed')
    ax.plot(df['disclosure_rate'], df['category'], 'o', color='lightblue', label='Disclosure rate', markersize=3)
    ax.set_xlim(0, 100)
    for i in range(len(df)): # Adding annotations to the bars
        ax.text(df['emissions_share'][i] + 1, i, f"{int(df['emissions_share'][i])}%", va='center', fontsize=5, color='black')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(loc='upper right', fontsize=5, frameon=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%')) # add % to x label
    fig.tight_layout()
    ax.grid(False)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig.savefig(os.path.join(figure_path, 's3_disclosure_by_category.png'), dpi=dpi)
    plt.close()