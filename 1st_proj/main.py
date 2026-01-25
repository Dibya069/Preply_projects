import sys
import argparse
from pathlib import Path
from components.logger import logger
from components.exception import CustomException
from server import YOLOv11Detector
from config import Config


# def parse_arguments():
#     """
#     Parse command line arguments.
    
#     Returns:
#         args: Parsed arguments
#     """
#     parser = argparse.ArgumentParser(description='YOLOv7 Real-time Object Detection')
    
#     parser.add_argument(
#         '--model',
#         type=str,
#         default='best.pt',
#         help='Path to the trained YOLOv7 model file (default: best.pt)'
#     )
    
#     parser.add_argument(
#         '--camera',
#         type=int,
#         default=0,
#         help='Camera device ID (default: 0 for default camera)'
#     )
    
#     parser.add_argument(
#         '--conf-threshold',
#         type=float,
#         default=0.25,
#         help='Confidence threshold for detections (default: 0.25)'
#     )
    
#     parser.add_argument(
#         '--iou-threshold',
#         type=float,
#         default=0.45,
#         help='IOU threshold for NMS (default: 0.45)'
#     )
    
#     parser.add_argument(
#         '--width',
#         type=int,
#         default=1280,
#         help='Display window width (default: 1280)'
#     )
    
#     parser.add_argument(
#         '--height',
#         type=int,
#         default=720,
#         help='Display window height (default: 720)'
#     )
    
#     return parser.parse_args()


def main():
    """
    Main function to run the YOLOv7 object detection pipeline.
    """
    try:
        logger.info("=" * 80)
        logger.info("YOLOv7 Real-time Object Detection Pipeline Started")
        logger.info("=" * 80)

        # Using configuration from Config
        # logger.info(f"Configuration:")
        # logger.info(f"  Model Path: {Config.MODEL_PATH}")
        # logger.info(f"  Camera ID: {Config.CAMERA_ID}")
        # logger.info(f"  Confidence Threshold: {Config.CONF_THRESHOLD}")
        # logger.info(f"  IOU Threshold: {Config.IOU_THRESHOLD}")
        # logger.info(f"  Display Size: {Config.DISPLAY_WIDTH}x{Config.DISPLAY_HEIGHT}")
        # logger.info(f"  Use GPU: {Config.USE_GPU}")

        # Check if model file exists
        model_path = Path(Config.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {Config.model_path}")
            logger.info("Please ensure your best.pt file path is correct in config_example.py")
            sys.exit(1)

        # Initialize YOLOv7 Detector
        detector = YOLOv11Detector()

        # Start camera detection
        logger.info("Starting real-time object detection...")
        logger.info("Press 'q' in the detection window to quit")

        detector.start_camera_detection(
            camera_id=Config.camera_id,
            display_width=Config.display_width,
            display_height=Config.display_height,
            bbox_thickness=Config.bbox_thickness,
            font_scale=Config.font_scale,
            font_thickness=Config.font_thickness
        )

        logger.info("=" * 80)
        logger.info("YOLOv7 Real-time Object Detection Pipeline Completed")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\nDetection interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()

