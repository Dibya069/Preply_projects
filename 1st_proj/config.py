## Here we are adding all the statics values

class Config:
    ## model config
    model_path= "/Users/dmohanty/Music/preply_proj/1st_proj/detect/train/weights/best.pt"
    confidence_threshold = 0.25
    iou_threshold = 0.5

    ## camera config
    camera_id = 0
    display_width = 1280 ## px
    display_height = 720 ## px
    cam_fps = 30

    ## detection display config
    bbox_thickness = 2
    font_scale = 1
    font_thickness = 2

    ## Performance config
    use_gpu = True

    ## logging config
    log_level = "INFO"

    ## color config
    custom_colors = None
