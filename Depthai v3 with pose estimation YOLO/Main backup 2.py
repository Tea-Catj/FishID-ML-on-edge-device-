import time
import depthai as dai
import cv2
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap

visualizer = dai.RemoteConnection(httpPort=8082)
fps_limit = 30

# Create pipeline
pipeline = dai.Pipeline()

with pipeline:
    
    #--------------------------------------------------------------------------------------------------------------------------
    # Define rgb cam and output
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)  # don’t forget .build()
    cameraOutput = camRgb.requestOutput((640, 320), type=dai.ImgFrame.Type.BGR888p, fps=fps_limit)

    #------------------------------------------------------------------------------------------------------------------------
    #define mono cam
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    leftOutput =  left.requestOutput((640, 320),type=dai.ImgFrame.Type.NV12, fps= fps_limit)
    rightOutput = right.requestOutput((640, 320),type= dai.ImgFrame.Type.NV12, fps = fps_limit)

    #define stereo 
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=leftOutput,
        right=rightOutput,
    )

    #stereo config
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    cameraOutput.link(stereo.inputAlignTo)

    #create depthmap
    depth_parser = pipeline.create(ApplyColormap).build(stereo.disparity)
    depth_parser.setMaxValue(int(stereo.initialConfig.getMaxDisparity())) # NOTE: Uncomment when DAI fixes a bug
    depth_parser.setColormap(cv2.COLORMAP_JET)


    nn_archive = dai.NNArchive(".\yolo11-nano-pose-estimation-exported-to-target-rvc2\yolo11n-pose.rvc2_legacy.rvc2.tar.xz")

    #---------------------------------------------------------------------------------------------------------
    # Create the neural network node
    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        cameraOutput, 
        nn_archive,
    )
    nn_with_parser.input.setBlocking(False)
    nn_with_parser.input.setMaxSize(1)
    
    #since the ParsingNeuralNetwork node is already choose the correct parser for the model, to mannually change the config
    # you need to get the parser and change its parameter  
    if nn_with_parser.getParser():
        parser = nn_with_parser.getParser()
        parser.setConfidenceThreshold(0.8)
        parser.setIouThreshold(0.5)

    #-------------------------------------------------------------------------------------------------------------
    #create pipeline for spatial calculation
    Spatial_data = pipeline.create(dai.node.SpatialLocationCalculator)

    #config for spatial
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 10
    config.depthThresholds.upperThreshold = 10000
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN

    #------------------------------------------------------------------------------------------------------------
    # Configure the visualizer node
    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Detections", nn_with_parser.out, "detections")
    visualizer.addTopic("Depth", depth_parser.out, "images")
    visualizer.addTopic("Left", leftOutput, "images")
    visualizer.addTopic("Right", rightOutput, "images")

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define queue

    #create neuro network parsed ouput queue for getting the model output
    parser_output_queue = nn_with_parser.out.createOutputQueue(maxSize= 1 ,blocking= False)
    keypoint_names=["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder","Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

    #create spatial data queue
    Spatial_data_queue = Spatial_data.out.createOutputQueue(maxSize= 1 ,blocking= False)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #start the pipeline
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while True:

        msg = parser_output_queue.tryGet() 
        
        if msg is not None:
            # Check for the existence of the 'detections' attribute
            if hasattr(msg, 'detections') and msg.detections:
                print(f"\n--- Found {len(msg.detections)} People ---")
                
                # 1. Iterate over each detected person
                for i, detection in enumerate(msg.detections):
                    print(f"{detection.label_name} {i+1}: Confidence: {detection.confidence:.2f}")
                    
                    # The keypoints are typically stored in a 'keypoints' attribute 
                    # within the detection object itself for pose models.
                    if hasattr(detection, 'keypoints') and detection.keypoints:
                        print(f"  Keypoints ({len(detection.keypoints)}):")
                        
                        # 2. Iterate over each keypoint for the current person
                        # The length of keypoint_names (17) should match len(detection.keypoints)
                        for kp_idx, keypoint in enumerate(detection.keypoints):
                            
                            # Get the name from your predefined list
                            name = keypoint_names[kp_idx]
                            
                            # Print the keypoint name and its normalized coordinates (0.0 to 1.0)
                            # NOTE: Keypoint objects usually have 'x' and 'y' or 'x_coord'/'y_coord'
                            print(f"    - {name}: ({keypoint.x:.4f}, {keypoint.y:.4f})")
                            

            # If the message is a different structure but has keypoints at the root level:
            elif hasattr(msg, 'keypoints'):
                print("--- Received Keypoints Message (Root Level) ---")
                # You would handle this structure if the parser returns keypoints directly,
                # but for YOLO pose, they are usually nested within detections.

        time.sleep(0.01)

        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
