import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds_[col] = Q1 - self.factor * IQR
                self.upper_bounds_[col] = Q3 + self.factor * IQR
        else:
            X_arr = np.asarray(X)
            for i in range(X_arr.shape[1]):
                col_data = X_arr[:, i]
                Q1 = np.nanpercentile(col_data, 25)
                Q3 = np.nanpercentile(col_data, 75)
                IQR = Q3 - Q1
                self.lower_bounds_[i] = Q1 - self.factor * IQR
                self.upper_bounds_[i] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else np.array(X, copy=True)
        if isinstance(X_out, pd.DataFrame):
            for col in X_out.columns:
                if col in self.lower_bounds_:
                    X_out[col] = X_out[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        else:
            for i in range(X_out.shape[1]):
                if i in self.lower_bounds_:
                    X_out[:, i] = np.clip(X_out[:, i], a_min=self.lower_bounds_[i], a_max=self.upper_bounds_[i])
        return X_out
