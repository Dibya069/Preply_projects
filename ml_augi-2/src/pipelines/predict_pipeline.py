import os
import sys
from src.logger import logging
from src.exception import CustomException
from flask import request

from src.utils import load_object

class PredictPipelines:
    def __init__(self, **kwargs):
        self.request = kwargs.get("request")

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")
            label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Load label encoder if it exists
            if os.path.exists(label_encoder_path):
                label_encoder = load_object(label_encoder_path)
                logging.info("Label encoder loaded successfully")
            else:
                label_encoder = None
                logging.warning("Label encoder not found, will return numeric predictions")

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            logging.info(f"Raw prediction: {pred}, dtype: {pred.dtype}")

            # Convert numeric predictions back to labels if label encoder exists
            if label_encoder is not None:
                # Convert predictions to integers before inverse_transform
                pred_int = pred.astype(int)
                pred_labels = label_encoder.inverse_transform(pred_int)
                logging.info(f"Predictions converted from {pred_int} to {pred_labels}")
                return pred_labels

            return pred

        except Exception as e:
            raise CustomException(e, sys)
            