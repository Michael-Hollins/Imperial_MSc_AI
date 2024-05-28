import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import plot_remove_labels

# Set parameters
sns.set_style("whitegrid")
data_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/data'
figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300

######################################################

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