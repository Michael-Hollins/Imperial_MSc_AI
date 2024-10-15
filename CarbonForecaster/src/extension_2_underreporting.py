# Underreporting exploration
import numpy as np
import pandas as pd
import pickle
from ml_pipeline import *
import os

figure_path = 'C:/Users/micha/OneDrive/Documents/AI_MSc/thesis/Imperial_MSc_AI/CarbonForecaster/report/figures'
plot_width_inch = 3   
plot_height_inch = 2   
dpi = 300
year_lower = 2016
year_upper = 2022
y_choice = ['s3_cat_total_intensity']
x_choice = ['mcap', 'revenue', 'capex', 'intangible_assets', 'lt_debt', 's1_and_s2_co2e_intensity']
id_choice = ['instrument']

# Step 1: Load the data and the tuned model
all_data = pd.read_csv('data/ftse_world_allcap_clean.csv')
XGB_params, optimal_boost_rounds = pickle.load(open('src/models/tuned/XGB_s3_cat_total_intensity_core_all_2016_2022_log.pickle', 'rb'))

# Step 2: Prepare the base dataframe
data = subset_to_date_range(all_data, year_lower=year_lower, year_upper=year_upper)
data = remove_missing_and_non_positive_values(data, y_choice)
data.reset_index(inplace=True)
material_obs = get_material_s3_category_reporting(data)

# Set out the variables in the X matrix
FY_dummies = [x for x in data.columns if str(x).startswith('FY_')]
econ_type_dummies = [x for x in data.columns if 'cc_classification_' in str(x)]
peer_dummies = [x for x in data.columns if 'business_sector_' in str(x)] # Use business sectors for ease
dummies = FY_dummies + econ_type_dummies + peer_dummies
x_choice_vars = x_choice + dummies

# Plot 1: Train on material, test on non-material

# Get the train/test split using log scales
training_data = data[material_obs] # We train on material observations
test_data = data[~material_obs] # We test on non-material observations
training_scaled = training_data.copy()
test_scaled = test_data.copy()
vars_to_scale = x_choice + y_choice # Log these variables, leave the dummies
for var in vars_to_scale:
    training_scaled[var] = np.log(training_scaled[var])
    test_scaled[var] = np.log(test_scaled[var])
    
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = \
        training_scaled[x_choice_vars].values, test_scaled[x_choice_vars].values, \
        training_scaled[y_choice].values, test_scaled[y_choice].values

# Step 3: Train on material obs
dtrain_scaled = xgb.DMatrix(X_train_scaled, label=y_train_scaled)
model = xgb.train(params=XGB_params, dtrain=dtrain_scaled, num_boost_round=optimal_boost_rounds)

# Step 4: Predict on non-material obs
dtest_scaled = xgb.DMatrix(X_test_scaled, label=y_test_scaled)
y_pred_scaled = model.predict(dtest_scaled)
y_pred_unscaled = np.exp(y_pred_scaled)

# Step 5: Plot results
log_XGB_predictions = y_pred_scaled
log_true_values = y_test_scaled

fig, ax = plt.subplots(figsize=(plot_width_inch*2, plot_height_inch*1.5), dpi=dpi)
plt.scatter(log_XGB_predictions, log_true_values, alpha=0.5, label='', s=5)

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
fig.savefig(os.path.join(figure_path, 's3_cat_material_predictions.png'))
plt.close()



