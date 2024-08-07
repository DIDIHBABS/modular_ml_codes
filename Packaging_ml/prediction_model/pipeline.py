from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.preocessing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np



classification_pipeline = Pipeline(
    [
        ("Drop Columns", pp.DropColumns(variable_to_drop=config.FEATURE_TO_DROP)),
        ("Input Mode",pp.InputMode(variables=config.CATEGORICAL_FEATURES),
        ("Input Mean", pp.InputMean(variables=config.NUM_FEATURES),
        ("Feature Transform", pp.FeatureTransform(variable_to_add=config.FEATURE_TO_ADD , variable_to_modify=config.FEATURE_TO_MODIFY)),
        ("Log Transform", pp.LogTransform(variable_to_transform=config.LOG_FEATURES)),
        ("Label Encoding", pp.LabelEncoding(variable_to_transform=config.CATEGORICAL_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)