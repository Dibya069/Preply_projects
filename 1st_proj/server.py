import cv2
import torch
import sys
import time
import numpy as np
from pathlib import Path
from components.logger import logger
from components.exception import CustomException
from components.utils import ModelUtils
from config import Config

# Try to import Ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available, will use PyTorch loading")


class YOLOv11Detector:
    """
    YOLOv11 Real-time Object Detection Server.
    """

    def __init__(self):
        """
        Initialize YOLOv7 detector.
        
        Args:
            model_path (str): Path to the trained model file (best.pt)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
        """
        try:
            self.model_path = Config.model_path
            self.conf_threshold = Config.confidence_threshold
            self.iou_threshold = Config.iou_threshold
            self.model = None
            self.device = None
            self.class_names = []
            self.is_ultralytics = False

            logger.info("Initializing YOLO Detector...")
            self._load_model()

        except Exception as e:
            raise CustomException(e, sys)
        
    def _load_model(self):
        """
        Load the YOLO model (supports both Ultralytics and PyTorch formats).
        """
        try:
            # Check if model file exists
            if not ModelUtils.check_model_exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Set device (GPU if available, else CPU)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")

            # Try to load with Ultralytics first (for YOLO11)
            if ULTRALYTICS_AVAILABLE:
                try:
                    logger.info(f"Loading model with Ultralytics from: {self.model_path}")
                    self.model = YOLO(self.model_path)
                    self.model.to(self.device)

                    # Get class names {"0":"person"}
                    self.class_names = self.model.names
                    if isinstance(self.class_names, dict):
                        self.class_names = list(self.class_names.values())

                    self.is_ultralytics = True
                    logger.info(f"Model loaded successfully with Ultralytics")
                    logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
                    logger.info(f"Classes: {self.class_names}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load with Ultralytics: {e}")
                    logger.info("Falling back to PyTorch loading...")

            # Fallback to PyTorch loading (for YOLOv11, custom models)
            logger.info(f"Loading model with PyTorch from: {self.model_path}")

            # Load the checkpoint (weights_only=False since this is your trained model)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Extract model from checkpoint, Torch works : [gpu, install gpu, load AI]
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'ema' in checkpoint:
                self.model = checkpoint['ema'].float()
            else:
                # If checkpoint is the model itself
                self.model = checkpoint

            # Convert to float and set to eval mode
            self.model = self.model.float()
            self.model.eval()
            self.model.to(self.device)

            # Set confidence and IOU thresholds
            if hasattr(self.model, 'conf'):
                self.model.conf = self.conf_threshold
            if hasattr(self.model, 'iou'):
                self.model.iou = self.iou_threshold

            # Get class names
            if hasattr(self.model, 'names'): ## For YOLO
                self.class_names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'): ## For Torch
                self.class_names = self.model.module.names
            else:
                # Default class names if not found
                self.class_names = [f'class{i}' for i in range(80)]
                logger.warning("Class names not found in model, using default names")

            self.is_ultralytics = False
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            logger.info(f"Classes: {self.class_names}")

        except Exception as e:
            raise CustomException(e, sys)
        
    def detect(self, frame):
        """
        Perform object detection on a frame.

        Args:
            frame: Input frame from camera

        Returns:
            detections: Detection results (numpy array with format: [x1, y1, x2, y2, confidence, class])
        """
        try:
            # Use Ultralytics inference if model is from Ultralytics
            if self.is_ultralytics:
                # Run inference with Ultralytics
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

                # Extract detections
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = results[0].boxes.conf.cpu().numpy()  # confidence
                    classes = results[0].boxes.cls.cpu().numpy()  # class

                    # Combine into single array
                    for i in range(len(boxes)):
                        detections.append([
                            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                            confs[i], classes[i]
                        ])

                return np.array(detections) if len(detections) > 0 else np.array([])

            # else:
            #     # PyTorch model inference (YOLOv11, custom models)
            #     h, w = frame.shape[:2]

            #     # Prepare image for inference
            #     img = self._prepare_image(frame)

            #     # Run inference
            #     with torch.no_grad():
            #         pred = self.model(img)

            #         # Handle different model output formats
            #         if isinstance(pred, tuple):
            #             pred = pred[0]

            #     # Apply NMS (Non-Maximum Suppression)
            #     pred = self._non_max_suppression(pred, self.conf_threshold, self.iou_threshold)

            #     # Process detections
            #     detections = []
            #     if pred[0] is not None and len(pred[0]) > 0:
            #         # Scale boxes back to original image size
            #         det = pred[0].cpu().numpy()

            #         # Scale coordinates
            #         img_h, img_w = img.shape[2:]
            #         det[:, [0, 2]] *= w / img_w
            #         det[:, [1, 3]] *= h / img_h

            #         detections = det

            #     return np.array(detections) if len(detections) > 0 else np.array([])

        except Exception as e:
            raise CustomException(e, sys)
        
    def start_camera_detection(self, camera_id, display_width, display_height, bbox_thickness, font_scale, font_thickness):
        """
        Start real-time object detection using camera.

        Args:
            camera_id (int): Camera device ID (0 for default camera)
            display_width (int): Display window width
            display_height (int): Display window height
            bbox_thickness (int): Thickness of bounding box lines
            font_scale (float): Font size for labels
            font_thickness (int): Font thickness for labels
        """
        try:
            logger.info(f"Starting camera detection with camera ID: {camera_id}")

            # Initialize camera
            cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                raise Exception(f"Failed to open camera with ID: {camera_id}")

            logger.info("Camera opened successfully")
            logger.info("Press 'q' to quit the detection")

            # FPS calculation variables
            prev_time = time.time()
            fps = 0

            while True:
                # Read frame from camera
                ret, frame = cap.read()

                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Perform detection
                detections = self.detect(frame)

                # Draw bounding boxes
                frame = ModelUtils.draw_bounding_boxes(frame, detections, self.class_names,
                                                      bbox_thickness, font_scale, font_thickness)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Add FPS to frame
                # frame = ModelUtils.add_fps_to_frame(frame, fps)
                
                # Add detection count
                # detection_count = len(detections)
                # cv2.putText(
                #     frame,
                #     f"Detections: {detection_count}",
                #     (10, 70),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 255, 0),
                #     2
                # )
                
                # Resize frame for display
                frame = cv2.resize(frame, (display_width, display_height))
                
                # Display frame
                cv2.imshow('YOLOv11 Real-time Object Detection', frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting detection...")
                    break
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera detection stopped successfully")
            
        except Exception as e:
            raise CustomException(e, sys)