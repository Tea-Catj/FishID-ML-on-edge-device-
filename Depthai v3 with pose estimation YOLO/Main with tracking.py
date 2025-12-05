import argparse
import time
import depthai as dai
import cv2
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap
from pathlib import Path
from model_utils import ensure_nn_archive
from fish_size_estimate import length_estimate, weight_estimate

# Import the tracker class (Make sure your tracking.py class is named PoseTrackerHandler)
from tracking import TrackerHandler 

def main(yes_visualizer: bool = True):

    fps_limit = 30
    # original/default archive path (relative to this script)
    nn_archive_path = ".\\yolo11-nano-pose-estimation-exported-to-target-rvc2\\yolo11n-pose.rvc2_legacy.rvc2.tar.xz"

    # Ensure NN archive path exists; this may prompt the user to provide/convert a .pt
    nn_archive_path = ensure_nn_archive(nn_archive_path, base_dir=Path(__file__).parent)
    print(f"Using NN archive at: {nn_archive_path}")

    visualizer = None
    if not yes_visualizer:
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
        
        # Mapping for your specific keypoints
        keypoint_names = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        keypoint_fish = ["Head", "Tail"]
        
        Spatial_data_queue = Spatial_cal.out.createOutputQueue(maxSize=1, blocking=False)
        Spatial_config_queue = Spatial_cal.inputConfig.createInputQueue(maxSize=1, blocking=False)
        stereo.depth.link(Spatial_cal.inputDepth)

        # --- TRACKER INITIALIZATION ---
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
                h, w, _ = color_image.shape
                
                if msg is not None:
                    try:
                        # --- TRACKING ---
                        # Pass raw detections to tracker
                        tracked_detections = tracker.get_tracked_results(msg.detections, color_image.shape)
                        
                        if tracked_detections.is_empty():
                            # print("No tracked fish found")
                            pass
                        else:
                            # print(f"Tracked {len(tracked_detections.xyxy)} fish")
                            
                            new_spatial_config = dai.SpatialLocationCalculatorConfig()
                            
                            # --- Iterate over TRACKED results ---
                            # supervision.Detections are arrays, so we iterate by index
                            for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                                x_min, y_min, x_max, y_max = xyxy
                                confidence = tracked_detections.confidence[i]
                                
                                # Draw Box & ID
                                label_text = f"ID {int(tracker_id)} | {confidence:.2f}"
                                cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                                cv2.putText(color_image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                # --- 3. Process Keypoints (if available) ---
                                # Check if keypoints exist on the tracked object
                                if tracked_detections.keypoints is not None and len(tracked_detections.keypoints) > i:
                                    
                                    # Get keypoints for this specific detection (Shape: [K, 2])
                                    fish_kps = tracked_detections.keypoints[i]
                                    
                                    # We only care about Head (0) and Tail (1)
                                    for kp_print_idx, (px, py) in enumerate(fish_kps):
                                        if kp_print_idx < len(keypoint_names):
                                            name = keypoint_names[kp_print_idx]
                                        else:
                                            name = f"KP{kp_print_idx+1}"
                                            
                                        # Only process valid pixels (0,0 often implies missing)
                                        if px == 0 and py == 0:
                                            continue

                                        # Draw keypoint
                                        cv2.circle(color_image, (px, py), radius=3, color=(0, 255, 0), thickness=-1)
                                        cv2.putText(color_image, name, (px + 5, py), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))

                                        # Prepare Spatial Config (ROI) for this keypoint
                                        # Convert pixel back to normalized (0-1) for DepthAI config
                                        norm_x = px / w
                                        norm_y = py / h
                                        
                                        roi_data = dai.SpatialLocationCalculatorConfigData()
                                        normalized_side = 0.02 # small box around point
                                        
                                        # Clamp ROI to be within 0-1
                                        roi_xmin = max(0, norm_x - normalized_side/2)
                                        roi_ymin = max(0, norm_y - normalized_side/2)
                                        roi_xmax = min(1, norm_x + normalized_side/2)
                                        roi_ymax = min(1, norm_y + normalized_side/2)

                                        roi_data.roi = dai.Rect(
                                            dai.Point2f(roi_xmin, roi_ymin),
                                            dai.Point2f(roi_xmax, roi_ymax),
                                        )
                                        roi_data.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                                        new_spatial_config.addROI(roi_data)
                                else:
                                    print(f"Fish ID {int(tracker_id)} has no keypoints") 
                            
                            # --- Send Spatial Config ---
                            if len(new_spatial_config.getConfigData()) > 0:
                                Spatial_config_queue.send(new_spatial_config)

                            # --- Retrieve Spatial Data ---
                            spatial_data = Spatial_data_queue.get().getSpatialLocations()
                            
                            if spatial_data:
                                spatial_idx = 0
                                # Iterate again to map spatial results back to specific tracks
                                for i, tracker_id in enumerate(tracked_detections.tracker_id):
                                    fish_kps = tracked_detections.keypoints[i]
                                    
                                    head_coord = None
                                    tail_coord = None
                                    
                                    # Loop through expected keypoints to grab their spatial data in order
                                    for kp_print_idx, (px, py) in enumerate(fish_kps):
                                        if kp_print_idx < len(keypoint_names):
                                            name = keypoint_names[kp_print_idx]
                                        else:
                                            name = f"KP{kp_print_idx+1}"
                                        if px == 0 and py == 0: continue
                                        
                                        if spatial_idx < len(spatial_data):
                                            sd = spatial_data[spatial_idx]
                                            spatial_idx += 1
                                            
                                            z = sd.spatialCoordinates.z
                                            y = sd.spatialCoordinates.y
                                            x = sd.spatialCoordinates.x
                                            
                                            if kp_print_idx == 0: # Head
                                                head_coord = (x, y, z)
                                            elif kp_print_idx == 1: # Tail
                                                tail_coord = (x, y, z)
                                    
                                    # Calculate Length for this specific Fish ID
                                    if head_coord and tail_coord:
                                        # Filter out bad Z values (e.g., 0 or extremely close)
                                        if head_coord[2] > 10 and tail_coord[2] > 10:
                                            length = length_estimate(head_coord, tail_coord)
                                            print(f"Fish {int(tracker_id)} Length: {length:.2f} mm")
                                        else:
                                            # print(f"Fish {int(tracker_id)}: Unreliable depth data")
                                            pass

                    except Exception as e:
                        print(f"Tracking error: {e}")

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
                    print("Visualizer waitKey failed; exiting loop")
                    break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Exception in main loop: {e}")
    finally:
        print("Cleaning up...")
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
    ap.add_argument('--yes-visualizer', action='store_false', help='Disable RemoteConnection visualizer (for testing)')
    args = ap.parse_args()
    main(yes_visualizer=args.yes_visualizer)