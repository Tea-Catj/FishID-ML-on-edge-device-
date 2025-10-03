from modelconverter import convert

#hubAPI key
HubAPI= 'tapi.oS4fjKC0F5hhchR2f5mZAQ.emKmFjyYtrjGzEV8coPfaMOTWXb-O-C-oh6UaL3ntHWZ21N65OFM4GWFeD0Y7lEiab8VxBVjqOKN8n43h6n8_A'

converted_model = convert.RVC2(
    api_key=HubAPI,
    path="yolo11n-pose.pt",
    number_of_shaves=5, 
    superblob=False,
    name="YOLO11 Nano Pose Estimation",
    description_short="Trained YOLO11 nano pose estimation model on COCO8 dataset.",
    yolo_version="yolov11",
    yolo_input_shape="640 320",
    yolo_class_names=["person"],
    tasks=["KEYPOINT_DETECTION"],
    license_type="MIT",
    is_public=False, 
)    