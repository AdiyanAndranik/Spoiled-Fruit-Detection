import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from spoiledDetection.logger import logging
from spoiledDetection.exception import AppException
from spoiledDetection.entity.config_entity import ModelTrainerConfig
from spoiledDetection.entity.artifacts_entity import ModelTrainerArtifact, DataIngestionArtifact
from spoiledDetection.constants import CLASS_NAMES
import sys

class PotentialsMethod:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances_squared = np.sum((self.X_train - x) ** 2, axis=1)
            potentials = 1 / (1 + self.alpha * distances_squared)
            class_potentials = []
            for cls in range(len(np.unique(self.y_train))):
                cls_mask = (self.y_train == cls)
                if np.sum(cls_mask) > 0:
                    class_potentials.append(np.mean(potentials[cls_mask]))
                else:
                    class_potentials.append(0.0)
            pred = np.argmax(class_potentials)
            predictions.append(pred)
        return np.array(predictions)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def train_models(self):
        try:
            logging.info("Starting model training")

            train_dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(self.data_ingestion_artifact.feature_store_path, 'train'),
                labels='inferred',
                label_mode='categorical',
                class_names=CLASS_NAMES,
                color_mode='rgb',
                batch_size=32,
                image_size=(64, 64),
                shuffle=True,
                seed=99,
            )
            test_dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(self.data_ingestion_artifact.feature_store_path, 'test'),
                labels='inferred',
                label_mode='categorical',
                class_names=CLASS_NAMES,
                color_mode='rgb',
                batch_size=32,
                image_size=(64, 64),
                shuffle=True,
                seed=99,
            )

            resize_rescale_layers = tf.keras.Sequential([
                tf.keras.layers.Resizing(64, 64),
                tf.keras.layers.Rescaling(1./255),
            ])

            def preprocess_image(image, label):
                image = resize_rescale_layers(image)
                return image, label

            training_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
            test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

            def process_dataset(dataset):
                features, labels = [], []
                for images, lbls in dataset:
                    for img, lbl in zip(images, lbls):
                        feat = img.numpy().flatten()
                        features.append(feat)
                        labels.append(np.argmax(lbl))
                return np.array(features), np.array(labels)

            X_train, y_train = process_dataset(training_dataset)
            X_test, y_test = process_dataset(test_dataset)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            potentials_model = PotentialsMethod(alpha=1.0)
            potentials_model.fit(X_train, y_train)
            y_pred_potentials = potentials_model.predict(X_test)
            accuracy_potentials = accuracy_score(y_test, y_pred_potentials)
            logging.info(f"Potentials Method - Accuracy: {accuracy_potentials:.4f}")

            svm_model = SVC()
            svm_model.fit(X_train, y_train)
            y_pred_svm = svm_model.predict(X_test)
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            logging.info(f"SVM - Accuracy: {accuracy_svm:.4f}")

            cnn_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            cnn_model.fit(training_dataset, epochs=5, validation_data=test_dataset, verbose=1)
            cnn_loss, cnn_accuracy = cnn_model.evaluate(test_dataset)
            logging.info(f"CNN - Accuracy: {cnn_accuracy:.4f}")

            accuracies = {
                'potentials': (accuracy_potentials, potentials_model),
                'svm': (accuracy_svm, svm_model),
                'cnn': (cnn_accuracy, cnn_model)
            }
            best_model_name = max(accuracies, key=lambda k: accuracies[k][0])
            best_accuracy, best_model = accuracies[best_model_name]
            logging.info(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            with open(self.model_trainer_config.potentials_model_path, 'wb') as f:
                pickle.dump(potentials_model, f)
            with open(self.model_trainer_config.svm_model_path, 'wb') as f:
                pickle.dump(svm_model, f)
            with open(self.model_trainer_config.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            cnn_model.save(self.model_trainer_config.cnn_model_path)

            if best_model_name == 'cnn':
                best_model_info = {'type': 'cnn', 'path': self.model_trainer_config.cnn_model_path}
            else:
                best_model_info = best_model
            with open(self.model_trainer_config.best_model_path, 'wb') as f:
                pickle.dump(best_model_info, f)
            logging.info(f"Best model saved as {self.model_trainer_config.best_model_path}")

            return ModelTrainerArtifact(
                potentials_model_path=self.model_trainer_config.potentials_model_path,
                svm_model_path=self.model_trainer_config.svm_model_path,
                cnn_model_path=self.model_trainer_config.cnn_model_path,
                scaler_path=self.model_trainer_config.scaler_path,
                best_model_path=self.model_trainer_config.best_model_path
            )
        except Exception as e:
            raise AppException(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model training")
            model_trainer_artifact = self.train_models()
            logging.info("Model training completed")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)