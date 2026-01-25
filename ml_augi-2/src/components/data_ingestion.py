import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys, os
from dataclasses import dataclass

## Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts', 'test.csv')
    raw_data_path=os.path.join('artifacts', 'raw.csv')
    label_encoder_path=os.path.join('artifacts', 'label_encoder.pkl')

## Creating data Ingestion calss
class DataIngestion:
    def __init__(self):
        self.ingestion_configuration = DataIngestionConfig()
        self.label_encoder = LabelEncoder()

    def get_data_outlier_settel(self, df, col_name):
        try:
            logging.info("Data_dealing with Outliers")
            column = df[col_name]
            z_scores = (column - column.mean()) / column.std()
            return z_scores
        
        except Exception as e:
            logging.info("Error Rasie from Data Ingestion (outlier) Stage")
            raise CustomException(e, sys)
        
##=====================================================================================================================

    def initiate_data_ingestioin(self, file_path=None):
        logging.info("Data Ingestion Mode START")

        try:
            # Load data from file
            if file_path:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format. Please use CSV or Excel files.")
                logging.info(f"Loaded data from file: {file_path}")
            else:
                # If no file provided, raise error
                raise ValueError("No file path provided. Please upload a CSV or Excel file for training.")

            # Convert 'Level' column from categorical to numeric using LabelEncoder
            if 'Level' in df.columns:
                df['Level'] = self.label_encoder.fit_transform(df['Level'])
                logging.info(f"Converted 'Level' column to numeric using LabelEncoder")
                logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

                # Save the label encoder for later use in predictions
                # from src.utils import save_obj
                # save_obj(
                #     file_path=self.ingestion_configuration.label_encoder_path,
                #     obj=self.label_encoder
                # )
                # logging.info(f"Label encoder saved at: {self.ingestion_configuration.label_encoder_path}")

            # Drop unnecessary columns if they exist
            columns_to_drop = ['index', 'Patient Id']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    logging.info(f"Dropped column: {col}")

            os.makedirs( os.path.dirname(self.ingestion_configuration.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_configuration.raw_data_path, index=False, header=True)
            logging.info("Raw Data is created")

            # Handle outliers for Age column if it exists
            if "Age" in df.columns:
                df["Age"] = self.get_data_outlier_settel(df, "Age")
                logging.info("Outliers dealing Completed for Age column")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv( self.ingestion_configuration.train_data_path, index=False, header=True )
            test_set.to_csv( self.ingestion_configuration.test_data_path, index=False, header=True )
            logging.info("train and test Data is created")

            return (
                self.ingestion_configuration.train_data_path,
                self.ingestion_configuration.test_data_path,
            )
        except Exception as e:
            logging.info("ERROR in DataIngestion Stage")
            raise CustomException(e, sys)