## check if the model present
## draw the bounding box
## getting the classes name with different color bounding box, confidence score
## resize the frame
## add the fps to the frame


import cv2
import numpy as np
import sys
from pathlib import Path

from components.exception import CustomException
from components.logger import logger


class ModelUtils:
    ## utility calss for model operation

    @staticmethod
    def check_model_exists(model_path):
        """
        check if the model exists
        """
        try:
            path = Path(model_path)

            if path.exists() and path.is_file():
                logger.info(f"model exists at: {model_path}")
                return True
            else:
                logger.info(f"model does not exists at: {model_path}") 
                return False
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def get_color_for_class(class_id):
        """
        get the color for the class
        """
        try:
            np.random.seed(class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            return color

        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def draw_bounding_boxes(frame, detections, class_names, bbox_thickness, font_scale, font_thickness):
        """
        draw the bounding box on the frame
        detections: 
        x1, y1, x2, y2, confidence, class_id

        class_names: list of class names ["anime", "fight", "victory"]
        """

        try:
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = detection[4]
                class_id = int(detection[5])

                ## class Name
                class_name = class_names[class_id]

                ## color for the class  
                color = ModelUtils.get_color_for_class(class_id)

                ## draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)

                ##preparing for label
                label = f"{class_name}: {confidence:.2f}"

                ## calculate the lable size and position
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                ##draw the label background
                cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)

                ## draw the label
                cv2.putText(frame, label, (x1, y1 - baseline + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            return frame
        except Exception as e:
            raise CustomException(e, sys) 
        
    @staticmethod
    def resize_frame(frame, width=640, height=640):
        """
        resize the frame
        """
        try:
            frame = cv2.resize(frame, (width, height))
            return frame
        except Exception as e:
            raise CustomException(e, sys)