import depthai as dai
import cv2
import depthai_nodes
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap, ImgFrameOverlay

visualizer = dai.RemoteConnection(httpPort=8082)
fps_limit = 30
# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)  # don’t forget .build()
cameraOutput = camRgb.requestOutput((640, 320), type=dai.ImgFrame.Type.BGR888p, fps=fps_limit)


#define mono cam
left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
leftOutput =  left.requestOutput((640, 320),type=dai.ImgFrame.Type.NV12, fps= fps_limit)
rightOutput = right.requestOutput((640, 320),type= dai.ImgFrame.Type.NV12, fps = fps_limit)

#create stereo 
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

# Create the neural network node
nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
    cameraOutput, 
    nn_archive
)
nn_with_parser.input.setBlocking(False)
nn_with_parser.input.setMaxSize(1)

#test
# # transform output array to colormap
# apply_colormap_node = pipeline.create(ApplyColormap).build(nn_with_parser.out)
# # overlay frames
# overlay_frames_node = pipeline.create(ImgFrameOverlay).build(
#     nn_with_parser.passthrough,
#     apply_colormap_node.out,
# )
# visualizer.addTopic("detections", overlay_frames_node.out, "images")

# Configure the visualizer node
visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
visualizer.addTopic("Detections", nn_with_parser.out, "detections")
visualizer.addTopic("Depth", depth_parser.out, "images")
visualizer.addTopic("Left", leftOutput, "images")
visualizer.addTopic("Right", rightOutput, "images")

#start the pipeline
pipeline.start()
visualizer.registerPipeline(pipeline)
while pipeline.isRunning():
    key = visualizer.waitKey(1)
    if key == ord("q"):
        print("Got t key from the remote connection!")
        break
