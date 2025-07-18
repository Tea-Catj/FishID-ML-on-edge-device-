
import cv2
import depthai as dai
import numpy as np
from calc import HostSpatialsCalc
from utility import *
import math
import time
from projector_3d import PointCloudVisualizer
import json
import datetime
import open3d as o3d

COLOR = True

lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

#get some info for depth calculation
baseline = 75 # in mm
focal_length= 426.55731201171875 # in pixel



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
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(median)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 400
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
    
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    stereo.depth.link(xoutDepth.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)  
    stereo.disparity.link(xoutDisp.input)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) 
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    camRgb.isp.link(xoutRBG.input)

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
            if len(synced) == 5:  # color, left, right, depth, nn
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
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        # depthDataQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        # rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        # rectifiedRightQueue=device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
        # depthDis= device.getOutputQueue(name="disp")
        # Color = device.getOutputQueue(name="color", maxSize=1, blocking=False)
        qs = []
        qs.append(device.getOutputQueue(name="depth", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue(name="disp"))
        qs.append(device.getOutputQueue(name="colored", maxSize=1, blocking=False))

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

        cv2.namedWindow("Disparity")

        text = TextHelper()
        hostSpatials = HostSpatialsCalc(device)
        y = 200
        x = 300
        step = 3
        delta = 5
        hostSpatials.setDeltaRoi(delta)

        print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")
       
        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False
        record = False
        write = False
        while True:
            for q in qs:
                new_msg = q.tryGet()
                if new_msg is not None:
                    msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:    
                    depthData = msgs["depth"]
                    spatials, centroid = hostSpatials.calc_spatials(depthData, (x,y)) # centroid == x/y in our case
                    
                    depthDataPoint = msgs["depth"].getFrame()
                    color2 = msgs['colored'].getCvFrame()

                    # Get the disparity map.
                    disparity1 = msgs["disp"].getFrame()
                    # Calculate spatial coordiantes from depth frame
                    
                    # Colormap disparity for display.
                    disparity2 = (disparity1 * disparityMultiplier).astype(np.uint8)
                    disparity3 = cv2.applyColorMap(disparity2, cv2.COLORMAP_JET)

                    # Get the left and right rectified frame.
                    leftFrame = msgs["rectifiedLeft"].getCvFrame()
                    rightFrame = msgs["rectifiedRight"].getCvFrame()

                    text.rectangle(disparity3, (x-delta, y-delta), (x+delta, y+delta))
                    text.putText(disparity3, "X: " + ("{:.4f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
                    text.putText(disparity3, "Y: " + ("{:.4f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
                    text.putText(disparity3, "Z: " + ("{:.4f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))


                    #get disparity frame data for mousecallback  
                    # frame_data_for_callback = (disparity3, disparity1)
                    # cv2.setMouseCallback("Disparity", mouse_callbackDepth, frame_data_for_callback)

                    #draw depth info on the diparity frame
                    # if depth is not None:
                    #     depth_text = f"{depth:.3f} m"  
                    #     disparity3 = cv2.circle(disparity3, (textX, textY), 2, (255, 255, 128), 2)
                    #     disparity3 = cv2.putText(disparity3, depth_text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                    

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
                
                        rgb = cv2.cvtColor(color2, cv2.COLOR_BGR2RGB)
                        pcl_converter.rgbd_to_projection(depthDataPoint, rgb)
                        pcl_converter.visualize_pcd()
                        
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
            # Quit when q is pressed
                break
            elif key == ord('w'):
                y -= step
            elif key == ord('a'):
                x -= step
            elif key == ord('s'):
                y += step
            elif key == ord('d'):
                x += step
            elif key == ord('r'): # Increase Delta
                if delta < 50:
                    delta += 1
                    hostSpatials.setDeltaRoi(delta)
            elif key == ord('f'): # Decrease Delta
                if 3 < delta:
                    delta -= 1
                    hostSpatials.setDeltaRoi(delta)
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
                o3d.io.write_point_cloud(f"{timestamp}.pcd", pcl_converter.pcl, compressed=True)
            elif key == ord ('j'):
                write = not write
                while write == True:
                    x = datetime.datetime.now()
                    timep = x.strftime("%X %p")
                    # Data to be written
                    dictionary = {
                                    "time": timep,
                                    "object": spatials['z']/1000,
                    }
                    # Serializing json
                    json_object = json.dumps(dictionary, indent=4)

                    # Writing to sample.json
                    with open("sample.json", "w") as outfile:
                        outfile.write(json_object)
                