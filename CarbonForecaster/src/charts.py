import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import plot_remove_labels
from eda import get_s3_prop_by_industry
from clean_raw_data import get_emissions_intensity_cols
from clean_raw_data import get_s3_cat_intensity_proportion_cols

# Set parameters
sns.set_style("whitegrid")
data_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/data'
figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/report/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300

REFRESH_PLOTS = False

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
    # Load the data
    data = pd.read_csv('data/ftse_global_allcap_clean.csv')

    # Subset date range
    df = data[data['year']== 2022].copy()

    # Initialise a list to store results
    res = list()
    # TODO: Revise this code for s3 intensity cols
    # Define some column names
    # observation_cols = ['instrument', 'year']
    # sector_col = 'icb_industry_name'
    # s3_cat_intensity_cols = get_emissions_intensity_cols(df)

    # df.loc[:, 'total_carbon_intensity'] = df.loc[:, s3_cat_intensity_cols].sum(axis=1)
    # df = df[df['total_carbon_intensity'] > 0] # Remove where we have no S3 category data
    # for col in s3_cat_intensity_cols:
    #     df.loc[f'{col}_proportion'] = df[col] / df['total_carbon_intensity']
    # cat_intensity_prop_cols = get_s3_cat_intensity_proportion_cols(df)
    # df = df[df[sector_col].notnull()] # remove where sectors are missing
    # sub = df[[sector_col, 'total_carbon_intensity']]
    # sub['total_carbon_intensity'] = np.log(sub['total_carbon_intensity'])

    # plt.figure(figsize=(plot_width_inch*2, plot_height_inch*2), dpi=dpi)
    # sns.scatterplot(data=sub, x=sub.index, y='total_carbon_intensity', hue='icb_industry_name', palette='tab10')
    # plt.title('Log Carbon Intensity by Industry')
    # plt.xlabel('')
    # plt.ylabel('Log Carbon Intensity')
    # plt.legend(title='Industry Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.ylim(0, sub['total_carbon_intensity'].max() * 1.1)
    # plt.show()
    
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
