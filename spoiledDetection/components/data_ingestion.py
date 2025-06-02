import os
import shutil
import yaml
import sys
from spoiledDetection.logger import logging
from spoiledDetection.exception import AppException
from spoiledDetection.entity.config_entity import DataIngestionConfig
from spoiledDetection.entity.artifacts_entity import DataIngestionArtifact
from spoiledDetection.constants import MERGED_DATASET_PATH, CLASS_NAMES

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            os.makedirs(self.data_ingestion_config.feature_store_file_path, exist_ok=True)

            for split in ['Train', 'Test']:
                src_dir = os.path.join(MERGED_DATASET_PATH, split)
                dst_dir = os.path.join(self.data_ingestion_config.feature_store_file_path, split.lower())
                os.makedirs(dst_dir, exist_ok=True)
                for class_name in CLASS_NAMES:
                    src_class_dir = os.path.join(src_dir, class_name)
                    dst_class_dir = os.path.join(dst_dir, class_name)
                    if not os.path.exists(src_class_dir):
                        logging.warning(f"Source directory {src_class_dir} does not exist")
                        continue
                    shutil.copytree(src_class_dir, dst_class_dir, dirs_exist_ok=True)
                    logging.info(f"Copied {src_class_dir} to {dst_class_dir}")

            data_yaml = {
                'train': os.path.join(self.data_ingestion_config.feature_store_file_path, 'train'),
                'val': os.path.join(self.data_ingestion_config.feature_store_file_path, 'test'),
                'nc': len(CLASS_NAMES),
                'names': CLASS_NAMES
            }
            with open(self.data_ingestion_config.data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            logging.info(f"Created data.yaml at {self.data_ingestion_config.data_yaml_path}")

            return DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_file_path,
                data_yaml_path=self.data_ingestion_config.data_yaml_path
            )
        except Exception as e:
            raise AppException(e, sys)