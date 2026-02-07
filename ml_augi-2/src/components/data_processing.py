
import pandas as pd
import numpy as np
import sys, os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass

from sklearn.impute import SimpleImputer ##Handle Missing value
from sklearn.preprocessing import StandardScaler    ## Handel Features scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTranformationConfig:
    preprocessro_obj_file_path = os.path.join("artifacts", "Preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

## ===================================================================================================================================

    def get_data_transformation_obj(self):
        try:
            logging.info("Initiate Data Transformation Scalling")
            ## Columns - All feature columns from cancer dataset (excluding index, Patient Id, and Level)
            col = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
                   'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
                   'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
                   'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                   'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                   'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']

            logging.info("Data Transformation Pipeline Initiated")
            pipe = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            ##Processing Pipeline
            Preprocess = ColumnTransformer([
                ('process', pipe, col)
            ])

            return Preprocess

        except Exception as e:
            logging.info("Error Rasie from Data Transformation (Scalling) Stage")
            raise CustomException(e, sys)
        
## ===================================================================================================================================

    def initiate_data_transformation_obj(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read Train and Test Completed")
            logging.info(f"The dataframe Head:  \n{train_df.head().to_string()}")
            logging.info(f"The dataframe Head:  \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_col = 'Level'
            drop_col = [target_col]

            ## dividing dataset into dependent and independent
            ## Train data
            input_feature_train_df = train_df.drop(columns=drop_col)
            traget_feature_train_df = train_df[target_col]

            ## Test data
            input_feature_test_df = test_df.drop(columns=drop_col)
            traget_feature_test_df = test_df[target_col]

            ## Data Transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(traget_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(traget_feature_test_df)]

            save_obj(
                file_path = self.data_transformation_config.preprocessro_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Applying Preprocessing object on training and test datasets.")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessro_obj_file_path
            )

        except Exception as e:
            logging.info("Error Rasie from Data Transformation (Final) Stage")
            raise CustomException(e, sys)
