import os
import sys
import pandas as pd
from dataclasses import dataclass
from flask import request

from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils


# =========================
# Config
# =========================
@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, "model.pkl")
    preprocessor_path: str = os.path.join(artifact_folder, "preprocessor.pkl")
    prediction_file_path: str = os.path.join(
        prediction_output_dirname, prediction_file_name
    )


# =========================
# Prediction Pipeline
# =========================
class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    # -------------------------------------------------
    # Save uploaded CSV
    # -------------------------------------------------
    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            if "file" not in self.request.files:
                raise Exception("No file part in the request")

            input_csv_file = self.request.files["file"]

            if input_csv_file.filename == "":
                raise Exception("No file selected for upload")

            file_name = os.path.basename(input_csv_file.filename)
            pred_file_path = os.path.join(pred_file_input_dir, file_name)

            input_csv_file.save(pred_file_path)
            logging.info(f"File saved at: {pred_file_path}")

            return pred_file_path

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    def predict(self, features: pd.DataFrame):
        try:
            model = self.utils.load_object(
                self.prediction_pipeline_config.model_file_path
            )
            preprocessor = self.utils.load_object(
                self.prediction_pipeline_config.preprocessor_path
            )

            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # Prediction dataframe
    # -------------------------------------------------
    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            input_dataframe = pd.read_csv(input_dataframe_path)

            # Drop unwanted index column
            if "Unnamed: 0" in input_dataframe.columns:
                input_dataframe.drop(columns=["Unnamed: 0"], inplace=True)

            
            # Remove target column if user uploads training-like data
            if TARGET_COLUMN in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(columns=[TARGET_COLUMN])

            predictions = self.predict(input_dataframe)

            # Add prediction column
            prediction_df = pd.read_csv(input_dataframe_path)

            target_column_mapping = {0: "bad", 1: "good"}
            prediction_df[TARGET_COLUMN] = [
                target_column_mapping[p] for p in predictions
            ]

            os.makedirs(
                self.prediction_pipeline_config.prediction_output_dirname,
                exist_ok=True,
            )

            prediction_df.to_csv(
                self.prediction_pipeline_config.prediction_file_path,
                index=False,
            )

            logging.info("Prediction completed successfully")

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------
    # Run pipeline
    # -------------------------------------------------
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            return self.prediction_pipeline_config

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)
