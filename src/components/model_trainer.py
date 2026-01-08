import sys
import os
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


# =========================
# Config
# =========================
@dataclass
class ModelTrainerConfig:
    artifact_folder: str = os.path.join(artifact_folder)
    trained_model_path: str = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy: float = 0.45
    model_config_file_path: str = os.path.join("config", "model.yaml")


# =========================
# Model Trainer
# =========================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(probability=True),
            "RandomForestClassifier": RandomForestClassifier(),
        }

    # =====================================================
    # Evaluate all models (ALL SCORES)
    # =====================================================
    def evaluate_models(self, X, y, models):

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            report = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                report[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                }

            return report

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================
    # Fine-tune best model
    # =====================================================
    def finetune_best_model(
        self,
        best_model_object,
        best_model_name,
        X_train,
        y_train,
    ):
        try:
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name}: {best_params}")

            return best_model_object.set_params(**best_params)

        except Exception as e:
            raise CustomException(e, sys)

    # =====================================================
    # Main Trainer Method
    # =====================================================
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # -------------------------------
            # Evaluate all models
            # -------------------------------
            model_report = self.evaluate_models(
                X=X_train,
                y=y_train,
                models=self.models
            )

            print("MODEL REPORT:")
            print(model_report)

            # -------------------------------
            # BEST MODEL SELECTION
            # Priority:
            # recall > f1 > precision > accuracy
            # -------------------------------
            best_model_name = None
            best_score_tuple = (-1, -1, -1, -1)  # recall, f1, precision, accuracy

            for model_name, scores in model_report.items():

                current_score = (
                    scores["recall"],
                    scores["f1"],
                    scores["precision"],
                    scores["accuracy"],
                )

                if current_score > best_score_tuple:
                    best_score_tuple = current_score
                    best_model_name = model_name

            best_scores = model_report[best_model_name]
            best_model = self.models[best_model_name]

            logging.info(
                f"Best model selected: {best_model_name} "
                f"with scores {best_scores}"
            )

            # -------------------------------
            # Fine-tuning
            # -------------------------------
            best_model = self.finetune_best_model(
                best_model_object=best_model,
                best_model_name=best_model_name,
                X_train=X_train,
                y_train=y_train,
            )

            # -------------------------------
            # Final training & evaluation
            # -------------------------------
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            final_accuracy = accuracy_score(y_test, y_pred)

            logging.info(
                f"Final accuracy of best model ({best_model_name}): {final_accuracy}"
            )

            if final_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    "No model met the expected accuracy threshold"
                )

            # -------------------------------
            # Save best model
            # -------------------------------
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True,
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
