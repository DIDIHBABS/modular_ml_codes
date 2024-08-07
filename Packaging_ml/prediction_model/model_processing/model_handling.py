import joblib
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config



# Serialization
def save_model(pipeline_to_save):
    save_path = os.path.join(config.SAVED_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")


# Deserialization 
def load_model(pipeline_to_load):
    saved_model_path = os.path.join(config.SAVED_MODEL_PATH, config.MODEL_NAME)
    load_model = joblib.load(pipeline_to_load)
    return load_model