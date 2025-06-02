from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path: str
    data_yaml_path: str

@dataclass
class DataValidationArtifact:
    status_file_path: str

@dataclass
class ModelTrainerArtifact:
    potentials_model_path: str
    svm_model_path: str
    cnn_model_path: str
    scaler_path: str
    best_model_path: str