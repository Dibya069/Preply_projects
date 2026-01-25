import sys, os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataTransformation
from src.components.model_building import ModelTrainig

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self, file_path=None):
        try:
            # Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestioin(file_path=file_path)
            logging.info(f"Train data: {train_data_path}")
            logging.info(f"Test data: {test_data_path}")

            # Data preprocessing
            logging.info("Step 2: Data Processing")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation_obj(
                train_data_path,
                test_data_path
            )
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Model building and Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainig()
            model_trainer.initiate_model_training(train_arr, test_arr)

        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    ## Run the pipeline
    pipeline = ModelTrainingPipeline()
    pipeline.run_pipeline(file_path="/Users/dmohanty/Music/preply_proj/ml_augi-2/upload/cancer patient data sets.csv")
    # print(f"\nFinal Model Score: {score}")