import os
import sys
from PIL import Image
from spoiledDetection.logger import logging
from spoiledDetection.exception import AppException
from spoiledDetection.entity.config_entity import DataValidationConfig
from spoiledDetection.entity.artifacts_entity import DataValidationArtifact, DataIngestionArtifact
from spoiledDetection.constants import CLASS_NAMES

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def validate_dataset(self) -> bool:
        try:
            logging.info("Starting data validation")
            validation_status = True
            status_messages = []

            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)

            for split in ['train', 'test']:
                split_dir = os.path.join(self.data_ingestion_artifact.feature_store_path, split)
                if not os.path.exists(split_dir):
                    validation_status = False
                    status_messages.append(f"{split} directory missing")
                    continue

                for class_name in CLASS_NAMES:
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.exists(class_dir):
                        validation_status = False
                        status_messages.append(f"{class_name} directory missing in {split}")
                        continue

                    images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if len(images) == 0:
                        validation_status = False
                        status_messages.append(f"No valid images in {class_name} directory of {split}")
                        continue

                    for img_name in images:
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            img = Image.open(img_path)
                            img.verify()
                            img.close()
                        except Exception as e:
                            validation_status = False
                            status_messages.append(f"Corrupted image: {img_path}")

            with open(self.data_validation_config.status_file_path, 'w') as f:
                if validation_status:
                    f.write("Validation status: True")
                else:
                    f.write("Validation status: False\n" + "\n".join(status_messages))
            logging.info(f"Validation status written to {self.data_validation_config.status_file_path}")

            return validation_status

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating data validation")
            self.validate_dataset()
            return DataValidationArtifact(
                status_file_path=self.data_validation_config.status_file_path
            )
        except Exception as e:
            raise AppException(e, sys)