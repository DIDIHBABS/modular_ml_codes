import os
import pathlib
import prediction_model


PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")


MODEL_NAME = 'classification.pkl'
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')


TRAIN_DATASET = 'train.csv'
TEST_DATASET ='test.csv'

TARGET = 'Loan_Status'
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = ['CoapplicantIncome']
FEATURE_TO_DROP = ['CoapplicantIncome']

NUM_FEATURES = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term']

CATEGORICAL_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'Credit_History', 'Property_Area']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']