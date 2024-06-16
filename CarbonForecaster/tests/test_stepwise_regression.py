import pytest 
from src.custom_functions import forward_step
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing

# Sample data for testing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

class TestForwardStep:
    def test_forward_step_adds_feature(self):
        included_features = set()
        inclusion_threshold = 0.05
        verbose = False

        new_features = forward_step(y, X, included_features, inclusion_threshold, verbose)
        
        assert len(new_features) > 0, "No features were added, but at least one was expected."
        
    def test_forward_step_feature_significance(self):
        included_features = set()
        inclusion_threshold = 0.05
        verbose = False

        new_features = forward_step(y, X, included_features, inclusion_threshold, verbose)
        
        candidate_features = set(X.columns) - included_features
        expanded_features = list(included_features) + list(new_features)
        model = sm.OLS(y, sm.add_constant(X[expanded_features])).fit()
        p_values = model.pvalues
        
        for feature in new_features:
            assert p_values[feature] < inclusion_threshold, f"Feature {feature} was added, but its p-value {p_values[feature]} is above the inclusion threshold."

    def test_forward_step_no_insignificant_features(self):
        included_features = set(X.columns)
        inclusion_threshold = 0.01
        verbose = False

        new_features = forward_step(y, X, included_features, inclusion_threshold, verbose)
        
        assert new_features == included_features, "No new features should be added when all features are already included."

if __name__ == "__main__":
    pytest.main()