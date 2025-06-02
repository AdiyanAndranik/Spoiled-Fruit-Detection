import os

ARTIFACTS_DIR: str = "artifacts"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
MERGED_DATASET_PATH: str = "D:/Projects/DiplomMar/merged_dataset/"
CLASS_NAMES: list = ['Fresh', 'Rotten']

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_STATUS_FILE: str = "status.txt"

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
POTENTIALS_MODEL_FILE: str = "potentials_model.pkl"
SVM_MODEL_FILE: str = "svm_model.pkl"
CNN_MODEL_FILE: str = "cnn_model.h5"
SCALER_FILE: str = "scaler.pkl"
BEST_MODEL_FILE: str = "best_model.pkl"