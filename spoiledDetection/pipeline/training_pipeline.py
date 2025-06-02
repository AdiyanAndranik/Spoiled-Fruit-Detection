import sys
from spoiledDetection.logger import logging
from spoiledDetection.exception import AppException
from spoiledDetection.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from spoiledDetection.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact
from spoiledDetection.components.data_ingestion import DataIngestion
from spoiledDetection.components.data_validation import DataValidation
from spoiledDetection.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Completed data ingestion")
            return data_ingestion_artifact
        except Exception as e:
            raise AppException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Completed data validation")
            return data_validation_artifact
        except Exception as e:
            raise AppException(e, sys)

    def start_model_training(self, data_ingestion_artifact: DataIngestionArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Entered the start_model_training method of TrainPipeline class")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info("Completed model training")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_training(data_ingestion_artifact)
            logging.info("Training pipeline completed")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)