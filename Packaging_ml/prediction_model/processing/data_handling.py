import os
import pandas as pd
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config



def load_dataset(file_path):
    url = os.path.join(config.DATAPATH,file_path)
    data = pd.read_csv(url)
    return data

