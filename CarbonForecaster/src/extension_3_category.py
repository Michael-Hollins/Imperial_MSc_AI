# Exploring category-level modelling

from ml_pipeline import *
import itertools
import os
import seaborn as sns
import matplotlib.patches as mpatches
from clean_raw_data import *
sns.set_style("whitegrid")
data_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/data'
figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/report/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300

plot_s3_distributions = False
plot_y_choice = False
plot_x_choice = False
plot_obs_choice = False
plot_modelling = False
outlier_analysis = False
SAVE_ALL_PREDICTIONS = False

if plot_s3_distributions:
    # Begin by comparing the distribution over all S3 categories
    data = pd.read_csv('data/ftse_world_allcap_clean.csv')
    s3_cat_cols = get_s3_cat_cols(data)
    intensity_cols = [x+'_intensity' for x in s3_cat_cols] + ['s3_cat_total_intensity']
    data = data[intensity_cols]
    new_column_names = {
        's3_purchased_goods_cat1_intensity': 'cat1',
        's3_capital_goods_cat2_intensity': 'cat2',
        's3_fuel_energy_cat3_intensity': 'cat3',
        's3_transportation_cat4_intensity': 'cat4',
        's3_waste_cat5_intensity': 'cat5',
        's3_business_travel_cat6_intensity': 'cat6',
        's3_employee_commuting_cat7_intensity': 'cat7',
        's3_leased_assets_cat8_intensity': 'cat8',
        's3_distribution_cat9_intensity': 'cat9',
        's3_processing_products_cat10_intensity': 'cat10',
        's3_use_of_sold_cat11_intensity': 'cat11',
        's3_EOL_treatment_cat12_intensity': 'cat12',
        's3_leased_assets_cat13_intensity': 'cat13',
        's3_franchises_cat14_intensity': 'cat14',
        's3_investments_cat15_intensity': 'cat15',
        's3_cat_total_intensity': 'total'
    }
    data = data.rename(columns=new_column_names)
    data = pd.melt(data)
    data = data.dropna()
    data['value'] = np.log(data['value'])

    # Define the correct order for the categories
    category_order = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'total']

    # Ensure the variable column follows this order
    data['variable'] = pd.Categorical(data['variable'], categories=category_order, ordered=True)



    # Set the plot style (no gridlines)
    sns.set(style="white")

    # Define the correct order for the categories
    category_order = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'total']

    # Ensure the variable column follows this order
    data['variable'] = pd.Categorical(data['variable'], categories=category_order, ordered=True)

    # Create a new column that stores the count of observations for each variable
    data_counts = data.groupby('variable').size().reset_index(name='count')

    # Create the FacetGrid for distribution plots with the correct category order, sharing the x-axis
    g = sns.FacetGrid(data, col="variable", col_wrap=4, sharex=True, sharey=False, height=3, col_order=category_order)

    # Map the distribution plot to the grid
    g.map(sns.histplot, "value", kde=True)

    # Add the titles with a gray ribbon and remove the default title
    for ax, (variable, count) in zip(g.axes.flat, data_counts.itertuples(index=False, name=None)):
        # Create the title with a gray ribbon (this is the title you want to keep)
        ax.text(0.5, 1.05, f"{variable}, n={count}",
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes, 
                fontsize=10, 
                fontweight='bold',
                bbox=dict(facecolor='gray', alpha=0.5, edgecolor='none'))

    # Remove the default titles that are behind the gray ribbon
    g.set_titles("")

    # Remove y-axis labels and ticks for all plots
    for ax in g.axes.flat:
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.spines['left'].set_visible(False)

    # Remove gridlines from all subplots
    for ax in g.axes.flat:
        ax.grid(False)  # Remove gridlines

    # Adjust the layout to ensure everything is visible and prevent tick values from being cut off
    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=3)  # Adjust padding to fit x-tick labels and plots

    # Show the plot
    plt.show()

    g.fig.savefig(os.path.join(figure_path, 's3_intensity_distribution.png'))


# Get all results

def generate_param_combinations():
    # Define the base lists for combinations
    y_choice_list = [['s3_business_travel_cat6'], ['s3_business_travel_cat6_intensity']]
    x_choice_str_list = ['extended', 'core', 'limited']
    observation_filter_list = [None, 'covid']
    
    # Initialize an empty list to store all parameter combinations
    param_combinations = []

    # Loop over all combinations of y_choice, x_choice_str, and observation_filter
    for y_choice, x_choice_str, observation_filter in itertools.product(y_choice_list, x_choice_str_list, observation_filter_list):
        # Set scaler_choice based on x_choice_str
        if x_choice_str == 'extended':
            scaler_choice = RobustScaler()
        else:
            scaler_choice = None
        
        # Set n_folds and min_firms_per_fold based on observation_filter
        if observation_filter == 'material':
            n_folds = 7
            min_firms_per_fold = 2
        else:
            n_folds = 8
            min_firms_per_fold = 3
        
        # Create a dictionary of the parameters for this combination
        param_combination = {
            'y_choice': y_choice,
            'x_choice_str': x_choice_str,
            'scaler_choice': scaler_choice,
            'observation_filter': observation_filter,
            'n_folds': n_folds,
            'min_firms_per_fold': min_firms_per_fold
        }

        # Append this combination to the list
        param_combinations.append(param_combination)

    return param_combinations

param_combination = generate_param_combinations()

# Fixed parameters
year_lower = 2016
year_upper = 2022
seed_choice = 42
tuned_path = 'src/models/tuned/'

if SAVE_ALL_PREDICTIONS:
    for params in param_combination:
        y_choice = params['y_choice']
        x_choice_str = params['x_choice_str']
        scaler_choice = params['scaler_choice']
        observation_filter = params['observation_filter']
        min_firms_per_fold = params['min_firms_per_fold']
        n_folds = params['n_folds']

        # Begin by preparing the data for mdelling
        data_train_scaled, data_test_scaled, data_test, X_train_scaled, X_test_scaled, x_choice, x_scaled, y_train_scaled, y_test_scaled, y_test, custom_folds_for_training_data, scaler_str, observation_filter_str = \
            prepare_data_pipeline(
                data_file_path='data/ftse_world_allcap_clean.csv', 
                y_choice=y_choice, 
                x_choice_str=x_choice_str, 
                scaler_choice=scaler_choice, 
                year_lower=year_lower, 
                year_upper=year_upper, 
                observation_filter=observation_filter, 
                min_firms_per_fold=min_firms_per_fold, 
                n_folds=n_folds, 
                seed_choice=seed_choice
            )

        # Peer group median
        data_train_peers_scaled = data_train_scaled.copy()
        peer_dummies = [x for x in data_train_peers_scaled.columns if 'peer_group_' in str(x)]
        data_train_peers_scaled = data_train_peers_scaled[y_choice + peer_dummies]
        data_train_peers_scaled['peer_group'] = data_train_peers_scaled[peer_dummies].idxmax(axis=1).str.replace('peer_group_', '')
        peer_averages_train = data_train_peers_scaled.groupby(['peer_group'])[y_choice].median().reset_index()
        peer_averages_train.columns = ['peer_group', 'peer_group_median']

        data_test_peers_scaled = data_test_scaled.copy()
        data_test_peers_scaled = data_test_peers_scaled[y_choice + peer_dummies]
        data_test_peers_scaled['peer_group'] = data_test_peers_scaled[peer_dummies].idxmax(axis=1).str.replace('peer_group_', '')
        data_test_peers_scaled = data_test_peers_scaled.merge(peer_averages_train, how='left', on='peer_group')

        MEDIAN_y_pred_scaled = np.array(data_test_peers_scaled['peer_group_median'])
        MEDIAN_y_pred_unscaled = unscale_predictions(MEDIAN_y_pred_scaled, X_scaled_data=X_test_scaled, x_choice_vars=x_choice,
                                                    scaler=scaler_choice, x_choice_str=x_choice_str, x_to_unscale=x_scaled)
        MEDIAN_res = get_results(y_test, MEDIAN_y_pred_unscaled)

        # OLS
        reg_OLS = LinearRegression().fit(X_train_scaled, y_train_scaled)
        OLS_y_pred_scaled = reg_OLS.predict(X_test_scaled)
        OLS_y_pred_unscaled = unscale_predictions(y_pred_scaled=OLS_y_pred_scaled, X_scaled_data=X_test_scaled, x_choice_vars=x_choice,
                                                    scaler=scaler_choice, x_choice_str=x_choice_str, x_to_unscale=x_scaled)
        OLS_res = get_results(y_test, OLS_y_pred_unscaled)

        # Load the other ML tuned models, fit them, and get the unscaled predictions
        EL_res, EL_y_pred_unscaled = get_model_results('EL', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled)

        QR_res, QR_y_pred_unscaled = get_model_results('QR', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled)

        MLP_res, MLP_y_pred_unscaled = get_model_results('MLP', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                    X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled)

        KNN_res, KNN_y_pred_unscaled = get_model_results('KNN', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                    X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled)

        dtrain_scaled = xgb.DMatrix(X_train_scaled, label=y_train_scaled)
        dtest_scaled = xgb.DMatrix(X_test_scaled, label=y_test)

        XGB_res, XGB_y_pred_unscaled = get_model_results('XGB', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                    X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled,
                                    dtrain_scaled=dtrain_scaled, dtest_scaled=dtest_scaled)

        RF_res, RF_y_pred_unscaled = get_model_results('RF', tuned_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str, 
                                X_train_scaled, y_train_scaled, X_test_scaled, y_test, x_choice, scaler_choice, x_scaled)

        # Collect predictions together and save
        dtest = data_test.copy()
        year_dummies = [x for x in dtest.columns if 'FY_' in str(x)]
        dtest['FY_' + str(year_lower)] = (dtest[year_dummies] == 0).all(axis=1).astype(int) # recover FY_2016 before changing back into year col
        year_dummies += ['FY_' + str(year_lower)]
        dtest['year'] = pd.from_dummies(dtest[year_dummies]).apply(lambda x: x.str.replace('FY_', ''))
        peer_dummies = [x for x in dtest.columns if 'peer_group_' in str(x)]
        dtest['peer_group'] = dtest[peer_dummies].idxmax(axis=1).str.replace('peer_group_', '')
        dtest = dtest[['instrument', 'year', 'revenue', 'mcap', 'peer_group'] + y_choice]
        dtest['IFM'] = MEDIAN_y_pred_unscaled
        dtest['OLS'] = OLS_y_pred_unscaled
        dtest['QR'] = QR_y_pred_unscaled
        dtest['EL'] = EL_y_pred_unscaled
        dtest['MLP'] = MLP_y_pred_unscaled
        dtest['KNN'] = KNN_y_pred_unscaled
        dtest['XGB'] = XGB_y_pred_unscaled
        dtest['RF'] = RF_y_pred_unscaled

        model_initials_list = ['IFM', 'OLS', 'QR', 'EL', 'MLP', 'KNN', 'XGB', 'RF']

        for model in model_initials_list:
            dtest[f'{model}_PE'] = ((dtest[model] - dtest[y_choice[0]]) / dtest[y_choice[0]]) * 100

        results_filename = y_choice[0] + '_' + x_choice_str + '_' + observation_filter_str + '_' + str(year_lower) + '_' + str(year_upper) + '_' + scaler_str + '.csv'

        dtest.to_csv(f'src/results/predictions/{results_filename}', index=False)

def get_predictions_path(file_path, y_choice, x_choice_str, observation_filter_str, year_lower, year_upper, scaler_str):
    return file_path + y_choice[0] + '_' + x_choice_str + '_' + observation_filter_str + '_' + str(year_lower) + '_' + str(year_upper) + '_' + scaler_str + '.csv'

file_path = 'src/results/predictions/'
all_results = None

for params in param_combination:
    y_choice = params['y_choice']
    x_choice_str = params['x_choice_str']
    scaler_choice = params['scaler_choice']
    scaler_str = 'robust' if isinstance(scaler_choice, RobustScaler) else 'log'
    observation_filter = params['observation_filter']
    if observation_filter is None:
        observation_filter_str = 'all'
    elif observation_filter == 'covid':
        observation_filter_str = 'ExCovid'
    elif observation_filter == 'material':
        observation_filter_str = 'material'
    else:
        raise ValueError('Observation filter is invalid.')
    
    # Load the predictions
    pred = pd.read_csv(get_predictions_path(file_path=file_path, y_choice=y_choice, x_choice_str=x_choice_str,
                                            observation_filter_str=observation_filter_str, year_lower=year_lower,
                                            year_upper=year_upper, scaler_str=scaler_str))
    pred['y_choice'] = y_choice[0]
    pred['x_choice'] = x_choice_str
    pred['observations'] = observation_filter_str
    
    if all_results is None:
        all_results = pred.copy()
    else:
        all_results = pd.concat([all_results, pred], ignore_index=True)

all_results.to_csv('src/results/predictions/s3_business_travel_cat6_all.csv', index=False)


# Choice 1: Target Variable: Absolute vs Intensity
if plot_y_choice:
    # First, melt the percentage errors into model, PE cols
    id_vars = ['instrument', 'year', 'y_choice', 'x_choice', 'observations']
    percentage_errors = ['IFM_PE', 'OLS_PE', 'QR_PE', 'EL_PE', 'MLP_PE', 'KNN_PE', 'XGB_PE', 'RF_PE']
    results = all_results.melt(id_vars=id_vars, value_vars=percentage_errors, var_name='model', value_name='PE')
    results['model'] = results['model'].apply(lambda x: x.replace('_PE', ''))
    results['APE'] = results['PE'].abs()

    def calculate_iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    # Group by the relevant variables and compute the median APE and IQR
    summary = results.groupby(['y_choice', 'x_choice', 'observations', 'model'])['APE'].agg(
        median_APE='median',
        IQR=calculate_iqr
    ).reset_index()

    summary.sort_values(by='median_APE', inplace=True)

    # Define colors for each model
    model_colors = {
        'EL': 'blue',
        'IFM': 'green',
        'KNN': 'purple',
        'MLP': 'orange',
        'OLS': 'red',
        'QR': 'cyan',
        'RF': 'pink',
        'XGB': 'brown'
    }

    # Get unique x_choice and observations values
    x_choices = summary['x_choice'].unique()
    observations = summary['observations'].unique()

    # Create a figure with a grid of subplots
    n_rows = len(observations)
    n_cols = len(x_choices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width_inch * 3, plot_width_inch * 3))

    # Iterate through rows and columns
    for i, obs in enumerate(observations):
        for j, x_choice in enumerate(x_choices):
            # Filter data for the current combination
            data_subset = summary[(summary['x_choice'] == x_choice) & (summary['observations'] == obs)]
            
            # Get the corresponding axes
            ax = axes[i, j]
            
            # Merge intensity and non-intensity data
            non_intensity_data = data_subset[~data_subset['y_choice'].str.contains('_intensity')]
            intensity_data = data_subset[data_subset['y_choice'].str.contains('_intensity')]
            
            merged_data = pd.merge(non_intensity_data, intensity_data, on='model', suffixes=('_abs', '_int'))

            # Plot the arrows for this subset
            for _, row in merged_data.iterrows():
                ax.annotate('',
                            xy=(row['IQR_int'], row['median_APE_int']),
                            xytext=(row['IQR_abs'], row['median_APE_abs']),
                            arrowprops=dict(color=model_colors[row['model']], arrowstyle='->', lw=2))
            
            # Set axis limits based on data
            x_min = merged_data[['IQR_abs', 'IQR_int']].min().min() * 0.8
            x_max = merged_data[['IQR_abs', 'IQR_int']].max().max() * 1.2
            y_min = merged_data[['median_APE_abs', 'median_APE_int']].min().min() * 0.8
            y_max = merged_data[['median_APE_abs', 'median_APE_int']].max().max() * 1.2
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Log scales
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Remove grid lines
            ax.grid(False)

            # Remove ticks and labels but keep the spines visible
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)

            # Column titles
            if i == 0:
                ax.set_title(f"{x_choice}", fontsize=16, pad=5)
            
            # Row titles on the right-hand side
            if j == n_cols - 1:
                ax.text(1.05, 0.5, f"{obs}", va='center', ha='left', rotation=270, fontsize=16, transform=ax.transAxes)

    # Global axis labels closer to the plot
    fig.text(0.5, 0.07, 'APE IQR', ha='center', fontsize=16)
    fig.text(0.07, 0.5, 'Average APE', va='center', rotation='vertical', fontsize=16)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.3, wspace=0.3)

    # Move the legend below the plot and remove the legend title
    handles = [plt.Line2D([], [], color=model_colors[model], lw=3) for model in model_colors.keys()]
    fig.legend(handles, model_colors.keys(), loc='lower center', ncol=len(model_colors), bbox_to_anchor=(0.5, 0.02), frameon=False)

    # Show the plot
    fig.savefig(os.path.join(figure_path, 's3_business_travel_cat6_y_choice.png'))
    plt.close()


# Choice 2: Explanatory Variables: Limited vs Core vs Extended
if plot_x_choice:
    intensities = all_results[all_results['s3_business_travel_cat6_intensity'].notnull()] # subset given the exerciser above
    percentage_errors = ['IFM_PE', 'OLS_PE', 'QR_PE', 'EL_PE', 'MLP_PE', 'KNN_PE', 'XGB_PE', 'RF_PE']

    results = intensities.melt(id_vars=['instrument', 'year', 'y_choice', 'x_choice', 'observations'], 
                            value_vars=percentage_errors, var_name='model', value_name='PE')

    results['APE'] = results['PE'].abs()

    summary = results.groupby(['observations', 'model', 'x_choice'])['APE'].agg(
        median_APE='median',
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index()

    summary.sort_values(by='median_APE', inplace=True)

    x_choice_colors = {
        'core': 'blue',
        'extended': 'green',
        'limited': 'red'
    }

    models = summary['model'].str.replace('_PE', '').unique()
    observations = summary['observations'].unique()

    n_rows = len(observations)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.7 * plot_width_inch * n_cols, 0.7 * plot_width_inch * n_rows), sharex=False, sharey=False)

    # Iterate through rows and columns for the subplots (switched)
    for i, obs in enumerate(observations):  # Rows for observations
        for j, model in enumerate(models):  # Columns for models
            ax = axes[i, j]  # Access the correct subplot

            # Filter data for the current model and observation
            data_subset = summary[(summary['model'].str.replace('_PE', '') == model) & (summary['observations'] == obs)]
            
            # Plot the scatter points for this subset, coloured by x_choice
            sns.scatterplot(
                x='IQR', 
                y='median_APE', 
                hue='x_choice', 
                palette=x_choice_colors,
                data=data_subset, 
                ax=ax,
                s=100,
                legend=False
            )
            
            # Dynamically adjust the axis limits based on the data in this subset
            x_min = data_subset['IQR'].min() * 0.9
            x_max = data_subset['IQR'].max() * 1.1
            y_min = data_subset['median_APE'].min() * 0.9
            y_max = data_subset['median_APE'].max() * 1.1
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Log scales
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Remove grid lines
            ax.grid(False)

            # Remove ticks and labels
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Ensure spines (outlines) are visible
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)
            
            # Set plot titles for columns (now models)
            if i == 0:
                ax.set_title(f"{model}", fontsize=18)
            
            # Set plot labels for rows (now observations)
            if j == 0:
                ax.set_ylabel(obs, fontsize=18)
                ax.tick_params(labelleft=True)  # Show y-axis labels only for the far-left column

    # Adjust text positions
    fig.text(0.5, 0.08, 'APE IQR', ha='center', fontsize=20)
    fig.text(0.05, 0.5, 'Median APE', va='center', rotation='vertical', fontsize=20,)  

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15)
    handles = [plt.Line2D([], [], color=x_choice_colors[x], marker='o', linestyle='', markersize=10) for x in x_choice_colors.keys()]
    fig.legend(handles, x_choice_colors.keys(), loc='lower center', ncol=len(x_choice_colors), frameon=False)

    # Save
    fig.savefig(os.path.join(figure_path, 's3_business_travel_cat6_x_choice.png'))
    plt.close()


# Choice 3: Observations: All vs ExCovid vs Material
if plot_obs_choice:
    core_intensities = all_results[(all_results['y_choice'] == 's3_business_travel_cat6_intensity') & (all_results['x_choice'] == 'core')]

    id_vars = ['instrument', 'year', 'y_choice', 'x_choice', 'observations']
    percentage_errors = ['IFM_PE', 'OLS_PE', 'QR_PE', 'EL_PE', 'MLP_PE', 'KNN_PE', 'XGB_PE', 'RF_PE']
    results = core_intensities.melt(id_vars=id_vars, value_vars=percentage_errors, var_name='model', value_name='PE')
    results['model'] = results['model'].apply(lambda x: x.replace('_PE', ''))
    results['APE'] = results['PE'].abs()

    def calculate_iqr(series):
        return series.quantile(0.75) - series.quantile(0.25)

    summary = results.groupby(['y_choice', 'x_choice', 'observations', 'model'])['APE'].agg(
        median_APE='median',
        IQR=calculate_iqr
    ).reset_index()

    summary.sort_values(by='median_APE', inplace=True)

    observation_colors = {
        'all': 'blue',
        'ExCovid': 'green',
        'material': 'orange'
    }

    # Create a new figure
    fig, ax = plt.subplots(figsize=(1.5*plot_width_inch, 1.5*plot_height_inch))

    sns.scatterplot(
                x='IQR', 
                y='median_APE', 
                hue='observations', 
                palette=observation_colors,
                data=summary, 
                ax=ax,
                s=100
            )

    # Add labels and title
    ax.set_xlabel('IQR', fontsize=9)
    ax.set_ylabel('Median APE', fontsize=9)
    ax.set_title('')

    # Show legend
    ax.legend(title='')
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_business_travel_cat6_obs_choice.png'))
    plt.close()

# Choice 4: The model
if not plot_modelling:
    def model_metrics(data):
        models = ['IFM', 'OLS', 'QR', 'EL', 'MLP', 'KNN', 'XGB', 'RF']
        results = {}
        
        # Loop over each model
        for model in models:
            predictions = data[model]
            true_values = data['s3_business_travel_cat6_intensity']
            absolute_percentage_errors = data[f'{model}_PE'].abs()
            
            # Calculate metrics
            medape = np.median(absolute_percentage_errors)  # Median APE
            iqr_ape = np.percentile(absolute_percentage_errors, 75) - np.percentile(absolute_percentage_errors, 25)  # IQR of APE
            ape_50 = (absolute_percentage_errors <= 50).mean() * 100  # Percentage of observations with APE <= 50%
            r_squared = r2_score(true_values, predictions)  # R-squared
            
            # Store metrics for the model
            results[model] = {
                'MDAPE': medape,
                'IQR_APE': iqr_ape,
                'PPAR': ape_50,
                'R-squared': r_squared
            }
        
        
        results_df = pd.DataFrame(results).T.round(2)
        return results_df  

    df = core_intensities = all_results[(all_results['y_choice'] == 's3_business_travel_cat6_intensity') &
                                        (all_results['x_choice'] == 'core') &
                                        (all_results['observations'] == 'ExCovid')]

    metrics_df = model_metrics(df)
    print(metrics_df)
    # Plot actuals vs predictions

    best_setup = pd.read_csv('src/results/predictions/s3_business_travel_cat6_intensity_core_ExCovid_2016_2022_log.csv')

    # Let's inspect the XGB outliers
    log_true_values = np.log(best_setup['s3_business_travel_cat6_intensity'])
    log_XGB_predictions = np.log(best_setup['XGB'])
    log_OLS_predictions = np.log(best_setup['OLS'])

    fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*1.5), dpi=dpi)

    # Plot XGB predictions in blue
    plt.scatter(log_XGB_predictions, log_true_values, alpha=0.5, label='XGB', s=5, color='blue')

    # Plot OLS predictions in red
    plt.scatter(log_OLS_predictions, log_true_values, alpha=0.5, label='OLS', s=5, color='green')

    # Add the 45-degree dashed line
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min(), log_true_values.max()], 
            'r--', linewidth=0.75, label='45-degree line')

    # plus or minus 100% error lines
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min() + np.log(2), log_true_values.max() + np.log(2)], 
            'gray', linestyle='--', linewidth=0.25, label='+100% Error')
    plt.plot([log_true_values.min(), log_true_values.max()], 
            [log_true_values.min() - np.log(2), log_true_values.max() - np.log(2)], 
            'gray', linestyle='--', linewidth=0.25, label='-100% Error')

    # Add axis labels
    plt.xlabel('Log S3 cat 6 intensity predicted', fontsize=9)
    plt.ylabel('Log S3 cat 6 intensity actual', fontsize=9)

    # Add legend for XGB and OLS
    plt.legend(loc='upper left', frameon=False, ncol=1, fontsize=7)

    # Show plot with tight layout
    plt.tight_layout()
    fig.savefig(os.path.join(figure_path, 's3_business_travel_cat6_intensity_core_ExCovid_2016_2022_prediction_vs_actuals_XGB_vs_OLS.png'))
    plt.close()

    # Plot prediction buckets

    best_model = pd.read_csv('src/results/predictions/s3_business_travel_cat6_intensity_core_ExCovid_2016_2022_log.csv')

    IFM_APE = best_model['IFM_PE'].abs()
    OLS_APE = best_model['OLS_PE'].abs()
    QR_APE = best_model['QR_PE'].abs()
    EL_APE = best_model['EL_PE'].abs()
    MLP_APE = best_model['MLP_PE'].abs()
    KNN_APE = best_model['KNN_PE'].abs()
    XGB_APE = best_model['XGB_PE'].abs()
    RF_APE = best_model['RF_PE'].abs()

    ape = {'IFM': IFM_APE, 'OLS': OLS_APE, 'QR': QR_APE, 'EN': EL_APE, 'MLP': MLP_APE, 'KNN': KNN_APE, 'XGB': XGB_APE, 'RF': RF_APE}
    ape = pd.DataFrame(ape)

    bins = [-np.inf, 10, 20, 50, 100, 200, np.inf]
    labels = ['Less than 10%', 'Between 10% and 20%', 'Between 20% and 50%', 
            'Between 50% and 100%', 'Between 100% and 200%', 'More than 200%']

    # Create an empty DataFrame to store the proportions
    ape_proportion_df = pd.DataFrame(index=labels)

    # Loop through each model in the DataFrame
    for model in ape.columns:
        binned = pd.cut(ape[model], bins=bins, labels=labels)
        proportion = binned.value_counts(normalize=True).sort_index()
        ape_proportion_df[model] = proportion

    # Create the bar plot 
    with plt.rc_context({'font.size': 12}):
        ax = ape_proportion_df.T.plot(kind='barh', stacked=True, figsize=(10, 8))  # Adjust figsize here if needed
        plt.xlabel('Proportion of Observations')
        plt.title('')  # Optional title
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, frameon=False)
        ax.set_xlim((0, 1))
        ax.grid(False)
        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig(os.path.join(figure_path, 's3_business_travel_cat6_intensity_core_ExCovid_2016_2022_prediction_buckets.png'))
        plt.close() 
