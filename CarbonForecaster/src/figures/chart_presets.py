import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from figures.chart_presets import plot_remove_labels
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

# Plotting
def plot_remove_labels(plot):
    plot.set_xlabel("")
    plot.set_ylabel("")
    plot.set_title("")