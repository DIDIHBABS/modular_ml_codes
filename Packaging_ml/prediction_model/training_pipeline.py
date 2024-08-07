import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.model_handling import save_model
from prediction_model.pipeline as pipe


def perform_training():
    train_data = load_dataset(config.TRAIN_DATASET)
    train_y = train_data[config.TARGET].map({'N':0,'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_model(pipe.classification_pipeline)

if __name__ == '__main__':
    perform_training()