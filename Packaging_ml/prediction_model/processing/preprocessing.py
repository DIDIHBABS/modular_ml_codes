import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_drop=None):
        self.variable_to_drop = variable_to_drop

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns= self.variable_to_drop)
        return X


class InputMean(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col].filna(self.mean_dict[col], inplace=True )
        return X


class InputMode(BaseEstimator, TransformerMixin):
    def __init__(self, variables= None):
        self.variables = variables

    def fit(self, X, y=None):
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col], inplace=True)
        return X


class FeatureTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_add, variable_to_modify):
        self.variable_to_add = variable_to_add
        self.variable_to_modify = variable_to_modify

    def fit(self):
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.variable_to_modify:
            X[col] = X[col] + X[self.variable_to_add]
        return X




class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_transform =None):
        self.variable_to_transform = variable_to_transform

    def fit(self):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variable_to_transform:
            X[col] = np.log(X[col])
        return X


class LabelEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, variable_to_transform=None):
        self.variable_to_transform = variable_to_transform

    def fit(self, X, y=None):
        self.label_dict = {}
        for col in self.variable_to_transform:
            transform = X[col].value_counts().sort_values(asending=True).index
            self.label_dict[col] = {k:i for i, k in enumerate(t,0)}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variable_to_transform:
            X[col] = X[col].map(self.label_dict[col])


