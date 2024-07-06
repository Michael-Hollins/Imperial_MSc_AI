from sklearn.datasets import fetch_california_housing
from utilities import stepwise_selection
import pandas as pd
import statsmodels.api as sm

if __name__ == '__main__':  
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    important_features = list(stepwise_selection(y, X))
    print(sm.OLS(y, sm.add_constant(X[important_features])).fit().summary())