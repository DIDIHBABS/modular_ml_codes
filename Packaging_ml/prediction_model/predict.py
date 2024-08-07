import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.model_handling import load_model
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model

classification_pipeline = load_model(config.MODEL_NAME)
def get_prediction(input_data):
    data = pd.DataFrame(input_data)
    predictions = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(predictions == 1, 'Y', 'N')
    result = {"prediction": output}
    return result

if __name__ =='__main__':
    get_prediction()

