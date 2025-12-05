from modelconverter import convert

def convert_model(model_path, yolo_version):
    #hubAPI key
    HubAPI= 'tapi.oS4fjKC0F5hhchR2f5mZAQ.LBPt2-sBo1CyUra0sl2ucxx2JqsqpYkjHYFHARmWf7-_Q-oLl3BkVIx5F0FDo8oe9Dx32-pGc3EnV0aC-4aHdA'

    converted_model = convert.RVC2(
        api_key=HubAPI,
        path= model_path,
        number_of_shaves=4, 
        superblob=False,
        name="YOLO-Pose-Estimation-RVC2",
        description_short="yolo",
        yolo_version= yolo_version,
        yolo_input_shape="640 320",
        yolo_class_names=["snakehead"],
        tasks=["KEYPOINT_DETECTION"],
        license_type="MIT",
        is_public=False
    )
    return converted_model    