
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from utility import *
import math
import time
from projector_3d import PointCloudVisualizer
import json
import datetime
import open3d as o3d
import blobconverter
import argparse

COLOR = True
stepSize = 0.05
newConfig = False
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

#get some info for depth calculation
baseline = 75 # in mm
focal_length= 426.55731201171875 # in pixel


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='model/yolo11n_openvino_2022.1_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolo11n.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 8, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True


def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()
 
    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono.setFps(30)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono.setFps(30)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
 
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
 
    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(median)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 350
    config.postProcessing.thresholdFilter.maxRange = 200000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
 
    return stereo
def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

# def mouse_callbackDepth(event, x, y, flags, param):
#     global depth, display_frame, textY, textX
#     if event == cv2.EVENT_LBUTTONDOWN and param is not None:
#         display_frame, disparity_frame = param
#         disparity_value = disparity_frame[y, x]
#         depth = ((baseline*focal_length)/disparity_value)/1000.0
#         textX = x
#         textY = y
#         print(depth)
#         return depth, display_frame

       
if __name__ == '__main__':
    mouseX = 0
    mouseY = 640
    # depth = 0
    # textX = 0
    # textY = 0
    # Start defining a pipeline
    pipeline = dai.Pipeline()
 
    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)
 
    # Combine left and right cameras to form a stereo pair
    camRgb = pipeline.create(dai.node.ColorCamera)
    stereo = getStereoPair(pipeline, monoLeft, monoRight)
    spatialLocationDetect = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
    spatialLocationCalculator2 = pipeline.create(dai.node.SpatialLocationCalculator)
    
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")
    
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("detection")
    

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
 
    xoutRBG = pipeline.createXLinkOut()
    xoutRBG.setStreamName("colored")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")
 
    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")
 
    xoutDisp = pipeline.create(dai.node.XLinkOut)
    xoutDisp.setStreamName("disp")
    
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData.setStreamName("SpatialData")
    
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
    xinSpatialCalcConfig.setStreamName("SpatialConfig")
    
    xoutSpatialData2 = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData2.setStreamName("SpatialData2")
    
    xinSpatialCalcConfig2 = pipeline.create(dai.node.XLinkIn)
    xinSpatialCalcConfig2.setStreamName("SpatialConfig2")
    
    #cam config
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    camRgb.setPreviewSize(W, H)
    camRgb.setInterleaved(False)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1,3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    
    # Network specific settings
    spatialLocationDetect.setBoundingBoxScaleFactor(1)
    spatialLocationDetect.setDepthLowerThreshold(100)
    spatialLocationDetect.setDepthUpperThreshold(5000)
    spatialLocationDetect.setConfidenceThreshold(confidenceThreshold)
    spatialLocationDetect.setNumClasses(classes)
    spatialLocationDetect.setCoordinateSize(coordinates)
    spatialLocationDetect.setAnchors(anchors)
    spatialLocationDetect.setAnchorMasks(anchorMasks)
    spatialLocationDetect.setIouThreshold(iouThreshold)
    spatialLocationDetect.setBlobPath(nnPath)
    spatialLocationDetect.setNumInferenceThreads(2)
    spatialLocationDetect.input.setBlocking(False)

    #linking
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)  
    camRgb.isp.link(xoutRBG.input)
    
    stereo.depth.link(xoutDepth.input)
    stereo.disparity.link(xoutDisp.input)
    stereo.depth.link(spatialLocationDetect.inputDepth)
    camRgb.preview.link(spatialLocationDetect.input)
    spatialLocationDetect.passthrough.link(xoutRBG.input)
    spatialLocationDetect.out.link(xoutNN.input)
    spatialLocationDetect.passthroughDepth.link(xoutDepth.input)
    spatialLocationDetect.outNetwork.link(nnOut.input)
    
    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig) 

    spatialLocationCalculator2.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator2.inputDepth)
    spatialLocationCalculator2.out.link(xoutSpatialData2.input)
    xinSpatialCalcConfig2.out.link(spatialLocationCalculator2.inputConfig) 
    
    class HostSync:
        def __init__(self):
            self.arrays = {}

        def add_msg(self, name, msg):
            if not name in self.arrays:
                self.arrays[name] = []
            # Add msg to array
            self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

            synced = {}
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if msg.getSequenceNum() == obj["seq"]:
                        synced[name] = obj["msg"]
                        break
            # If there are 5 (all) synced msgs, remove all old msgs
            # and return synced msgs
            if len(synced) == 7:  # color, left, right, depth, nn
                # Remove old msgs
                for name, arr in self.arrays.items():
                    for i, obj in enumerate(arr):
                        if obj["seq"] < msg.getSequenceNum():
                            arr.remove(obj)
                        else:
                            break
                return synced
            return False


    with dai.Device(pipeline) as device:
        SpatialConfig= device.getInputQueue(name="SpatialConfig", maxSize=1, blocking=False)
        SpatialDataI= device.getOutputQueue(name="SpatialData", maxSize=1, blocking=False)
        SpatialConfig2= device.getInputQueue(name="SpatialConfig2", maxSize=1, blocking=False)
        SpatialDataI2= device.getOutputQueue(name="SpatialData2", maxSize=1, blocking=False)
        qs = []
        qs.append(device.getOutputQueue(name="depth", maxSize=4, blocking=False))
        qs.append(device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="disp", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="colored", maxSize=4, blocking=False))
        qs.append(device.getOutputQueue(name="nn", maxSize=4, blocking=False))
        qs.append(device.getOutputQueue(name="detection", maxSize=4, blocking=False))
       
        calibData = device.readCalibration2()
        w, h = camRgb.getIspSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, dai.Size2f(w, h))
        pcl_converter = PointCloudVisualizer(intrinsics, w, h)
        sync = HostSync()
        leftFrame, rightFrame, disparity3, color2 = None, None, None, None

        # Calculate a multiplier for color mapping disparity map
        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()
 
        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallback)
        cv2.namedWindow("RGB")

        cv2.namedWindow("Disparity")

        color = (255, 255, 255)
       
        # Variable use to toggle between side by side view and one frame view.
        startTime = time.monotonic()
        sideBySide = False
        record = False
        write = False
        printOutputLayersOnce = True
        counter = 0
        fps = 0
        color = (255, 255, 255)

        while True:
            for q in qs:
                new_msg = q.tryGet()
                if new_msg is not None:
                    msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:    
                    # depthData = msgs["depth"]
                    inDet = msgs["detection"]       
                    inNN = msgs["nn"]
                    if printOutputLayersOnce:
                        toPrint = 'Output layer names:'
                        for ten in inNN.getAllLayerNames():
                            toPrint = f'{toPrint} {ten},'
                            print(toPrint)
                        printOutputLayersOnce = False
                    
                    #get depth frame and color frame
                    depthDataPoint = msgs["depth"].getFrame()
                    color2 = msgs['colored'].getCvFrame()

                    # Get the disparity map.
                    disparity1 = msgs["disp"].getFrame()
                    
                    
                    depth_downscaled = disparity1[::4]
                    if np.all(depth_downscaled == 0):
                        min_depth = 0  # Set a default minimum depth value when all elements are zero
                    else:
                        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                        max_depth = np.percentile(depth_downscaled, 99)

                    #FPS counter
                    counter+=1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1 :
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time


                    # Colormap disparity for display.
                    disparity2 = (disparity1 * disparityMultiplier).astype(np.uint8)
                    disparity3 = cv2.applyColorMap(disparity2, cv2.COLORMAP_JET)

                    # Get the left and right rectified frame.
                    leftFrame = msgs["rectifiedLeft"].getCvFrame()
                    rightFrame = msgs["rectifiedRight"].getCvFrame()
                    
                    # show spatial coordiantes on depth frame
                    height = color2.shape[0]
                    width  = color2.shape[1]
                    detections = inDet.detections
                    for detection in detections:
                        roiData = detection.boundingBoxMapping
                        roi = roiData.roi
                        roi = roi.denormalize(width=disparity3.shape[1], height=disparity3.shape[0])
                        
                        xmin = int(roi.topLeft().x)
                        print("xmin:", xmin)
                        ymin = int(roi.topLeft().y)
                        print("ymin", ymin)
                        xmax = int(roi.bottomRight().x)
                        print("xmax:", xmax)
                        ymax = int(roi.bottomRight().y)
                        print("Ymax:",  ymax)            
                        
                        fontType = cv2.FONT_HERSHEY_TRIPLEX
                        cv2.rectangle(disparity3, (xmin, ymin), (xmax, ymax), color, 1)
                        
                        # Denormalize bounding box
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        
                         # Config
                        # Example: One ROI at the top-left and one at the top-right of the detected object
                        # Adjust these based on the actual shape/orientation of your objects
                        roi1_x = int((x1 + (x2-x1)*0.02)) #int(x1 + (x2 - x1) / 2)  #  5% from top
                        roi1_y = int(y1 + (y2 - y1) / 2) #int((y1+(y2-y1)*0.05))  # half lep
                        roi2_x = int((x2 + (-x2 + x1)*0.02))  # 10% from top
                        roi2_y = int(y2 + (-y2 + y1) / 2)  # half  right
                        roi_size = 4

                        #normalized ROI
                        n_roi1_x = roi1_x /disparity3.shape[1]
                        n_roi1_y = roi1_y/disparity3.shape[0]
                        n_roi2_x = roi2_x /disparity3.shape[1]
                        n_roi2_y = roi2_y/disparity3.shape[0]

                        #integer ROI for visualize ???
                        i_roi1_x = int(roi1_x)
                        i_roi1_y = int(roi1_y)
                        i_roi2_x = int(roi2_x)
                        i_roi2_y = int(roi2_y)    

                        print(n_roi1_x )
                        print(n_roi1_y)
                        
                        cfg1 = dai.SpatialLocationCalculatorConfig()
                        cfg2 = dai.SpatialLocationCalculatorConfig()

                        config1 = dai.SpatialLocationCalculatorConfigData()
                        config1.roi = dai.Rect(dai.Point2f(n_roi1_x, n_roi1_x), dai.Point2f(n_roi1_y, n_roi1_y))
                        config1.depthThresholds.lowerThreshold = 100
                        config1.depthThresholds.upperThreshold = 10000
                        cv2.circle(disparity3, (i_roi1_x, i_roi1_y), 5 , color, 1)
                        cv2.circle(color2, (roi1_x, roi1_y),5 , color, 1)
                        cfg1.addROI(config1)
                        SpatialConfig.send(cfg1)

                        config2 = dai.SpatialLocationCalculatorConfigData()
                        config2.roi = dai.Rect(dai.Point2f(n_roi2_x, n_roi2_x), dai.Point2f((n_roi2_y), (n_roi2_y)))
                        config2.depthThresholds.lowerThreshold = 100
                        config2.depthThresholds.upperThreshold = 10000
                        cv2.circle(disparity3, (i_roi2_x, i_roi2_y), 5 , color, 1)
                        cv2.circle(color2, (roi2_x, roi2_y), 5 , color, 1)    
                        cfg2.addROI(config2)
                        SpatialConfig2.send(cfg2)
                        

                        roidepth = SpatialDataI.tryGet()
                        roidepth2 = SpatialDataI2.tryGet()
                        print(roidepth)
                        print(roidepth2)
                        if roidepth is not None and roidepth2 is not None:
                            DepthData1 = roidepth.getSpatialLocations()
                            for cor1 in DepthData1:
                                coords1x = cor1.spatialCoordinates.x
                                coords1y = cor1.spatialCoordinates.y
                                coords1z = cor1.spatialCoordinates.z
                                
                            DepthData2 = roidepth2.getSpatialLocations()
                            for cor2 in DepthData2:
                                coords2x = cor2.spatialCoordinates.x
                                coords2y = cor2.spatialCoordinates.y
                                coords2z = cor2.spatialCoordinates.z

                            # Calculate Euclidean distance (length)
                            length_m = np.sqrt((coords2x - coords1x)**2 + 
                                                (coords2y - coords1y)**2 + 
                                                (coords2z - coords1z)**2)
                            print(f"Object length: {length_m/1000:.3f} meters")
                            cv2.putText(color2, f"height: {length_m/1000:.3f} m", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        else:
                            cv2.putText(color2, f"height: none", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        try:
                            label = labels[detection.label]
                        except:
                            label = detection.label

                        cv2.putText(color2, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(color2, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(color2, f"X: {int(detection.spatialCoordinates.x)/1000} m", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(color2, f"Y: {int(detection.spatialCoordinates.y)/1000} m", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(color2, f"Z: {int(detection.spatialCoordinates.z)/1000} m", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        
                        cv2.rectangle(color2, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                        cv2.putText(color2, "NN fps: {:.2f}".format(fps), (2, color2.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                       
                    
                    if sideBySide:
                        # Show side by side view.
                        imOut = np.hstack((leftFrame, rightFrame))
                    else:
                        # Show overlapping frames.
                        imOut = np.uint8(leftFrame / 2 + rightFrame / 2)
                        # Convert to RGB.
                        imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)
                        # Draw scan line.
                        imOut = cv2.line(imOut, (mouseX, mouseY), (1280, mouseY), (0, 0, 255), 2)
                        # Draw clicked point.
                        imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (255, 255, 128), 2)
                        cv2.imshow("Stereo Pair", imOut)
                        cv2.imshow("Disparity", disparity3)
                        cv2.imshow("RGB", color2)

                        # rgb = cv2.cvtColor(color2, cv2.COLOR_BGR2RGB)
                        # pcl_converter.rgbd_to_projection(depthDataPoint, rgb)
                        # pcl_converter.visualize_pcd()
                
                    
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
            # Quit when q is pressed
                break

            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide    
            elif key == ord('x'):
                #start recording    
                if record == False:
                    # Define video writer parameters for disparity
                    fourcc_disparity = cv2.VideoWriter_fourcc(*'MJPG')
                    out_disparity = cv2.VideoWriter('test.avi', fourcc_disparity, 30.0, (640, 400)) # Adjust resolution
                    record = True
                elif record == True:
                    record = False
            if record:
                out_disparity.write(disparity3)
            elif key == ord('g'):
                #capture image
                timestamp = str(int(time.time()))
                cv2.imwrite(f"{timestamp}_depth.png", disparity3)
                cv2.imwrite(f"{timestamp}_rgb.png", color2)
            