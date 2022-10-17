import numpy as np
from numpy import NaN
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class TempVarMissingTransformer(BaseEstimator, TransformerMixin):
    """
    This class is used for the feature engine pipeline to transform temp vars with missing values
    """

    def __init__(self, temp_vars):

        if not isinstance(temp_vars, list):
            raise ValueError('variables should be a list')

        self.temp_vars = temp_vars

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        for var in self.temp_vars:
            X.loc[pd.to_datetime(X[var], utc=True, errors="coerce", dayfirst=True).isna(), var] = NaN
            X[var] = pd.to_datetime(X[var], dayfirst=True)

        return X


class TempVarElapsedTimeTransformer(BaseEstimator, TransformerMixin):
    """
    This class is used for the feature engine pipeline to create columns with elapsed time between datetimes
    """

    def __init__(self, name, var1, var2):

        if not isinstance(var1, str):
            raise ValueError('var1 should be a string')
        if not isinstance(var2, str):
            raise ValueError('var2 should be a string')
        if not isinstance(name, str):
            raise ValueError('name should be a string')

        self.var1 = var1
        self.var2 = var2
        self.name = name

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        X[self.name] = (X[self.var1] - X[self.var2]).dt.seconds / 60
        X[self.name].astype("Int64")

        X.drop([self.var1, self.var2], axis=1, inplace=True)
        return X


class TempVarSplitTransformer(BaseEstimator, TransformerMixin):
    """
    This class is used for the feature engine pipeline to split datetime columns into year, month, day, hour, minute, second
    """

    def __init__(self, vars, date_or_dt):

        if not isinstance(vars, list):
            raise ValueError('variables should be a list')
        if not isinstance(date_or_dt, str):
            raise ValueError('date_or_dt should be a string')

        self.vars = vars
        self.date_or_dt = date_or_dt

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        for var in self.vars:
            if self.date_or_dt == "date":
                X["year_" + str(var)] = X[var].dt.year
                X["month_" + str(var)] = X[var].dt.month
                X["day_" + str(var)] = X[var].dt.day
            else:
                X["year_" + str(var)] = X[var].dt.year
                X["month_" + str(var)] = X[var].dt.month
                X["day_" + str(var)] = X[var].dt.day
                X["hour_" + str(var)] = X[var].dt.hour
                X["minute_" + str(var)] = X[var].dt.minute
                X["second_" + str(var)] = X[var].dt.second

            X.drop([var], axis=1, inplace=True)

        return X
