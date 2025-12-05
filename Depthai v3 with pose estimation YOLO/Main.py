
import argparse
import time
import depthai as dai
import cv2
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap
from pathlib import Path
from model_utils import ensure_nn_archive
from fish_size_estimate import length_estimate, weight_estimate
from tracking import TrackerHandler

def main(no_visualizer: bool = True):
    fps_limit = 30
    # original/default archive path (relative to this script)
    nn_archive_path = ".\\yolo11-nano-pose-estimation-exported-to-target-rvc2\\yolo11n-pose.rvc2_legacy.rvc2.tar.xz"

    # Ensure NN archive path exists; this may prompt the user to provide/convert a .pt
    nn_archive_path = ensure_nn_archive(nn_archive_path, base_dir=Path(__file__).parent)
    print(f"Using NN archive at: {nn_archive_path}")

    visualizer = None
    if not no_visualizer:
        try:
            visualizer = dai.RemoteConnection(httpPort=8082)
            print("Visualizer enabled (http://localhost:8082)")
        except Exception as e:
            print(f"Failed to create visualizer: {e}")
            visualizer = None

    pipeline = dai.Pipeline()

    with pipeline:
        # Define rgb cam and output
        camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cameraOutput = camRgb.requestOutput((640, 320), type=dai.ImgFrame.Type.BGR888p, fps=fps_limit)

        # define mono cam
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        leftOutput = left.requestOutput((640, 320), type=dai.ImgFrame.Type.NV12, fps=fps_limit)
        rightOutput = right.requestOutput((640, 320), type=dai.ImgFrame.Type.NV12, fps=fps_limit)

        # define stereo
        stereo = pipeline.create(dai.node.StereoDepth).build(left=leftOutput, right=rightOutput)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)
        stereo.setLeftRightCheck(True)
        cameraOutput.link(stereo.inputAlignTo)

        # create depthmap
        depth_parser = pipeline.create(ApplyColormap).build(stereo.disparity)
        depth_parser.setColormap(cv2.COLORMAP_JET)

        nn_archive = dai.NNArchive(nn_archive_path)

        # Create the neural network node
        nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(cameraOutput, nn_archive)
        nn_with_parser.input.setBlocking(False)
        nn_with_parser.input.setMaxSize(1)

        
        parser = nn_with_parser.getParser()
        parser.setConfidenceThreshold(0.4)
        parser.setIouThreshold(0.5)

        # create pipeline for spatial calculation
        Spatial_cal = pipeline.create(dai.node.SpatialLocationCalculator)

        # Configure the visualizer node (only if requested)
        if visualizer is not None:
            try:
                visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
                visualizer.addTopic("Detections", nn_with_parser.out, "detections")
                visualizer.addTopic("Depth", depth_parser.out, "images")
                visualizer.addTopic("Left", leftOutput, "images")
                visualizer.addTopic("Right", rightOutput, "images")
            except Exception as e:
                print(f"Failed to configure visualizer topics: {e}")

        # define queues
        colorFrame = cameraOutput.createOutputQueue(maxSize=1, blocking=False)
        parser_output_queue = nn_with_parser.out.createOutputQueue(maxSize=1, blocking=False)
        keypoint_names = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        keypoint_fish = ["Head", "Tail"]
        Spatial_data_queue = Spatial_cal.out.createOutputQueue(maxSize=1, blocking=False)
        Spatial_config_queue = Spatial_cal.inputConfig.createInputQueue(maxSize=1, blocking=False)
        # Spatial_depth_queue = Spatial_cal.passthroughDepth.createOutputQueue(maxSize=1, blocking=False)
        stereo.depth.link(Spatial_cal.inputDepth)

        #initalized tracker
        tracker = TrackerHandler()

    # start the pipeline
    pipeline.start()
    if visualizer is not None:
        try:
            visualizer.registerPipeline(pipeline)
        except Exception as e:
            print(f"Failed to register pipeline with visualizer: {e}")

    try:
        while True:
            msg = parser_output_queue.get()
            color_msg = colorFrame.get()
            if color_msg is not None:
                color_image = color_msg.getCvFrame() 
                # get frame dimensions
                h, w, _ = color_image.shape
                
                if msg is not None:
                    if hasattr(msg, 'detections') and msg.detections:
                        print(f"\n--- Found {len(msg.detections)} fish(es) ---")

                        try:
                            tracked = tracker.get_tracked_results(msg.detections, color_image.shape)
                            print(f"Tracked results: {tracked.tracker_id}")

                        except Exception as e:
                            print(f"error: {e}")
                                

                        for fish_idx, detection in enumerate(msg.detections):
                            if fish_idx < len(tracked.tracker_id):
                                fishID = tracked.tracker_id[fish_idx]

                            print(f"{detection.label_name} {fish_idx+1} ID[{fishID}]: Confidence: {detection.confidence:.2f}")

                            if hasattr(detection, 'keypoints') and detection.keypoints:
                                print(f" Keypoints ({len(detection.keypoints)}):")
                                new_spatial_config = dai.SpatialLocationCalculatorConfig()
                                for kp_idx, keypoint in enumerate(detection.keypoints):
                                    roi_data = dai.SpatialLocationCalculatorConfigData()
                                    
                                    # unnormalize for cv2 drawing
                                    xi = int(keypoint.x*w)
                                    yi = int(keypoint.y*h)
                                    cv2.circle(color_image, (xi,yi), radius = 2, color=(0, 255, 0), thickness=-1)
                                    cv2.putText(color_image, keypoint_names[kp_idx], (xi + 10, yi), cv2.FONT_HERSHEY_TRIPLEX, 0.3, color=(0, 255, 0))
                                    
                                    center_x = keypoint.x
                                    center_y = keypoint.y
                                    normalized_side = 0.02
                                    roi_data.roi = dai.Rect(
                                        dai.Point2f(center_x - normalized_side/2, center_y - normalized_side/2),
                                        dai.Point2f(center_x + normalized_side/2, center_y + normalized_side/2),
                                    )
                                    new_spatial_config.addROI(roi_data)

                                Spatial_config_queue.send(new_spatial_config)

                                spatial_data = Spatial_data_queue.get().getSpatialLocations()
                                Head_coords = None
                                Tail_coords = None
                                if spatial_data is not None:
                                    for kp_print_idx, spatial_location in enumerate(spatial_data):
                                        if kp_print_idx < len(keypoint_names):
                                            name = keypoint_names[kp_print_idx]
                                        else:
                                            name = f"KP{kp_print_idx+1}"
                                        z = spatial_location.spatialCoordinates.z
                                        y = spatial_location.spatialCoordinates.y
                                        x = spatial_location.spatialCoordinates.x
                                        print(f"{name} {kp_print_idx+1} - Z: {z:.2f} mm, Y: {y:.2f}, X: {x:.2f}")

                                        # Estimate length only if Z values are reliable
                                        if z >= 3.4:
                                            if name == "Head":
                                                Head_coords = (x, y, z)
                                            if name == "Tail":
                                                Tail_coords = (x, y, z)
                                            if Head_coords is not None and Tail_coords is not None:
                                                length = length_estimate(Head_coords, Tail_coords)
                                                print(f"Estimated length: {length:.2f} mm")
                                        else:
                                            print(f"The Z value of coridinate too low for reliable measurement.")

                cv2.imshow("key point ye", color_image)
                

            key = cv2.waitKey(1)
            if key == ord("x"):
                print("Got x key from keyboard!")
                break

            # handle key input only if visualizer is present
            if visualizer is not None:
                try:
                    key = visualizer.waitKey(1)
                    if key == ord('q'):
                        print("Got q key from the remote connection!")
                        break
                except Exception:
                    # if visualizer fails, break out and cleanup
                    print("Visualizer waitKey failed; exiting loop")
                    break
    
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        print("Cleaning up: stopping pipeline and closing visualizer (if available)")
        try:
            pipeline.stop()
        except Exception:
            pass
        if visualizer is not None:
            if hasattr(visualizer, 'unregisterPipeline'):
                try:
                    visualizer.unregisterPipeline(pipeline)
                except Exception:
                    pass
            if hasattr(visualizer, 'close'):
                try:
                    visualizer.close()
                except Exception:
                    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-visualizer', action='store_false', help='Disable RemoteConnection visualizer (for testing)')
    args = ap.parse_args()
    main(no_visualizer=args.no_visualizer)
