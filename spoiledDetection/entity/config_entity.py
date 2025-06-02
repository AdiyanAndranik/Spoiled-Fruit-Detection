import os
from dataclasses import dataclass
from spoiledDetection.constants import *

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )
    data_yaml_path: str = os.path.join(
        feature_store_file_path, "data.yaml"
    )

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )
    status_file_path: str = os.path.join(
        data_validation_dir, DATA_VALIDATION_STATUS_FILE
    )
    feature_store_path: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME, DATA_INGESTION_FEATURE_STORE_DIR
    )

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME
    )
    potentials_model_path: str = os.path.join(
        model_trainer_dir, POTENTIALS_MODEL_FILE
    )
    svm_model_path: str = os.path.join(
        model_trainer_dir, SVM_MODEL_FILE
    )
    cnn_model_path: str = os.path.join(
        model_trainer_dir, CNN_MODEL_FILE
    )
    scaler_path: str = os.path.join(
        model_trainer_dir, SCALER_FILE
    )
    best_model_path: str = os.path.join(
        model_trainer_dir, BEST_MODEL_FILE
    )