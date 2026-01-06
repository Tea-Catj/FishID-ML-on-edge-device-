import argparse
import time
import depthai as dai
import cv2
import numpy as np
import queue
import threading
from depthai_nodes.node import ParsingNeuralNetwork, ApplyColormap
from pathlib import Path
from model_utils import ensure_nn_archive
from DataBaseHandler import DataBaseHandler
from fish_size_estimate import generate_intermediate_points, fit_polynomial_curve_and_calculate_length, length_estimate, fit_curve_and_calculate_length
from tracking import TrackerHandler 
from record import ThreadedVideoRecorder
from Monitor import SystemMonitor

def main(yes_visualizer: bool = False, httpPort: int = 8082, webSocketPort: int = 8765, no_cv2: bool = False):
    fps_limit = 20
    FRAME_TIMEOUT = 70
    MINIMAL_MOTION_THRESHOLD = 0.05
    LETHARGY_TIMEOUT_MINUTES = 0.5
    DEAD_TIMEOUT_MINUTES = 1 
    
    # original/default archive path
    nn_archive_path = ".\\yolo11-nano-pose-estimation-exported-to-target-rvc2\\yolo11n-pose.rvc2_legacy.rvc2.tar.xz"

    nn_archive_path = ensure_nn_archive(nn_archive_path, base_dir=Path(__file__).parent)
    print(f"Using NN archive at: {nn_archive_path}")

    frame_lock = threading.Lock()
    
    # Initialize monitor
    monitor = SystemMonitor()
    
    # ---------------------------------------------------------------------
    # --- DATA STRUCTURES FOR TRACKING ---
    # ---------------------------------------------------------------------

    last_known_positions_3d = {}  # Dict to store last known 3D positions for each fish
    distance_traveled = {}  # Dict to store total distance traveled for each fish
    last_active_time = {}  # Dict to store last active timestamp for each fish (TIME-BASED)
    first_seen_time = {}  # Dict to store when fish was first seen

    visualizer = None
    if yes_visualizer:
        try:
            visualizer = dai.RemoteConnection( address='0.0.0.0', 
                                              webSocketPort = webSocketPort , 
                                              serveFrontend=True, 
                                              httpPort = httpPort)
            print("Visualizer enabled (http://localhost:8082)")
        except Exception as e:
            print(f"Failed to create visualizer: {e}")
            visualizer = None

    #--------initialize video recorder-----------
    video_recorder = ThreadedVideoRecorder()
    recording_started = False
    
    #--------initialize pipeline-----------
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
        stereo.setExtendedDisparity(False)
        stereo.setLeftRightCheck(True)
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
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
        parser.setConfidenceThreshold(0.35)
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
        colorFrame = cameraOutput.createOutputQueue(maxSize=4, blocking=False)
        parser_output_queue = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)
        # Mapping for your specific keypoints
        keypoint_fish = ["Head", "Tail"]
        
        Spatial_data_queue = Spatial_cal.out.createOutputQueue(maxSize=4, blocking=False)
        Spatial_config_queue = Spatial_cal.inputConfig.createInputQueue(maxSize=4, blocking=False)
        stereo.depth.link(Spatial_cal.inputDepth)

        # --- TRACKER INITIALIZATION ---
        tracker = TrackerHandler()
        frame_count = 0  # Initialize frame counter
        total_unique_fish = 0  # Initialize total unique fish counter
        last_seen_frame = {}  # Dict to store last seen frame for each fish
        seen_track_ids = set()  # Set to store all seen fish IDs
        
        # --- Database INITIALIZATION ---
        DB = DataBaseHandler()
        DB.connect()
        
    # start the pipeline
    pipeline.start()
    if visualizer is not None:
        try:
            visualizer.registerPipeline(pipeline)
        except Exception as e:
            print(f"Failed to register pipeline with visualizer: {e}")

    last_db_update = {}
    status = 'active' # Default status
    
    try:
        while True:
            # Start monitoring this frame
            monitor.start_frame()
            current_time = time.time()  # Get current time for this frame
        
            try:
                msg = parser_output_queue.get(timeout=0.1)
                color_msg = colorFrame.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if color_msg:
                
                frame_count += 1
                color_image = color_msg.getCvFrame() 
                h, w, _ = color_image.shape
                live_fish_count = 0  # Initialize live fish count
                
                # Dictionary to store keypoints for each fish
                fish_keypoints_dict = {} 

                if msg:
                    try:
                        # --- TRACKING ---
                        # Pass raw detections to tracker
                        tracked_detections = tracker.get_tracked_results(msg.detections, color_image.shape)

                        if tracked_detections.is_empty():
                            print("No tracked fish found")          
                        else:
                            live_fish_count = len(tracked_detections.tracker_id) # Update live fish count 
                            # print(f"Tracked {live_fish_count} fish")
                            if live_fish_count > 0:
                                new_spatial_config = dai.SpatialLocationCalculatorConfig()
                                
                                # --- Iterate over TRACKED results ---
                                # supervision.Detections are arrays, so we iterate by index
                                for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                                    x_min, y_min, x_max, y_max = xyxy
                                    confidence = tracked_detections.confidence[i]

                                    # Update last seen frame for this fish 
                                    last_seen_frame[tracker_id] = frame_count
                                    if tracker_id not in seen_track_ids:  # If fish is new
                                        seen_track_ids.add(tracker_id)  # Add to seen IDs
                                        total_unique_fish = len(seen_track_ids)  # Update total unique fish count
                                        last_active_time[tracker_id] = current_time  # Set initial active time
                                        first_seen_time[tracker_id] = current_time  # Record when fish was first seen
                                    
                                    
                                    current_total_distance_m = DB.get_last_known_distance(tracker_id)  # Get last known distance
                                    dist_m = 0  # Initialize movement distance
                                    current_position_3d = None  # Initialize 3D position

                                    # --- Process Keypoints ---
                                    # Check if keypoints exist on the tracked object
                                    if tracked_detections.keypoints is not None and len(tracked_detections.keypoints) > i:
                                        
                                        # Get keypoints for this specific detection (Shape: [K, 2])
                                        fish_kps = tracked_detections.keypoints[i]
                                        head_2d = None
                                        tail_2d = None
                                        
                                        # First pass: extract head and tail
                                        for kp_print_idx, (px, py) in enumerate(fish_kps):
                                            if kp_print_idx < len(keypoint_fish):
                                                name = keypoint_fish[kp_print_idx]
                                                # Draw keypoint
                                                cv2.circle(color_image, (px, py), radius=3, color=(0, 255, 0), thickness=-1)
                                                cv2.putText(color_image, name, (px + 5, py), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0))
                                                
                                            # Store head and tail coordinates
                                            if kp_print_idx == 0:  # Head
                                                head_2d = (px, py)
                                            elif kp_print_idx == 1:  # Tail
                                                tail_2d = (px, py)
                                        
                                        # Generate intermediate points if we have both head and tail
                                        if head_2d and tail_2d:
                                            # Generate ALL points along the line (including head and tail at ends)
                                            all_keypoints_2d = generate_intermediate_points(head_2d, tail_2d, num_points=5)
                                            
                                            # print(f"Fish ID {int(tracker_id)} All Keypoints: {all_keypoints_2d}")
                                            
                                            # Store for later use
                                            fish_keypoints_dict[i] = all_keypoints_2d
                                            
                                            # Create ROIs for ALL points (head, tail, and intermediates)
                                            for kp_idx, point in enumerate(all_keypoints_2d):
                                                px, py = point
                                                
                                                # Only process valid pixels
                                                if px == 0 and py == 0:
                                                    continue
                                                
                                                # Prepare Spatial Config (ROI) for this keypoint
                                                norm_x = px / w
                                                norm_y = py / h
                                                
                                                roi_data = dai.SpatialLocationCalculatorConfigData()
                                                normalized_side = 0.02  # small box around point
                                                
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
                                            print(f"Fish ID {int(tracker_id)} missing head or tail keypoints")
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
                                        if i in fish_keypoints_dict:
                                            all_keypoints_2d = fish_keypoints_dict[i]
                                            # Collect 3D coordinates for all points
                                            spatial_points_3d = []
                                            head_coord = None
                                            tail_coord = None
                                            
                                            # Loop through expected keypoints to grab their spatial data in order
                                            for kp_idx, point in enumerate(all_keypoints_2d):
                                                px, py = point
                                                
                                                if spatial_idx < len(spatial_data):
                                                    sd = spatial_data[spatial_idx]
                                                    spatial_idx += 1
                                                    
                                                    z = sd.spatialCoordinates.z
                                                    y = sd.spatialCoordinates.y
                                                    x = sd.spatialCoordinates.x
                                                    
                                                    # Store the 3D point
                                                    point_3d = (x, y, z)
                                                    spatial_points_3d.append(point_3d)

                                                    if kp_idx == 0: # Head
                                                        head_coord = point_3d
                                                    elif kp_idx == len(all_keypoints_2d) - 1: # Tail
                                                        tail_coord = point_3d
                                            
                                            # Calculate info Length for this specific Fish ID
                                            if head_coord and tail_coord:
                                                
                                                # --- Curve-fitted length calculation ---
                                                # Filter out bad Z values (e.g., 0 or extremely close)
                                                valid_points = [p for p in spatial_points_3d if p[2] > 10]
                                                
                                                if len(valid_points) >= 3 and head_coord[2] > 10 and tail_coord[2] > 10:
                                                    # Calculate curve-fitted length
                                                    curve_length = fit_polynomial_curve_and_calculate_length(valid_points, degree=2)
                                                    # Also calculate Euclidean distance for comparison
                                                    euclidean_length = length_estimate(head_coord, tail_coord)
                                                    length = curve_length  # Use curve-fitted length
                                                    print(f"Fish {int(tracker_id)} Curve-fitted Length: {curve_length:.2f} mm, accuracy: {(curve_length/300)*100:.2f} %, euclidean: {euclidean_length:.2f} mm, accuracy: {(euclidean_length/300)*100:.2f} %")
                                                elif head_coord[2] > 10 and tail_coord[2] > 10:
                                                    # Fall back to Euclidean distance
                                                    length = length_estimate(head_coord, tail_coord)
                                                    print(f"Fish {int(tracker_id)} Euclidean Length: {length:.2f} mm")
                                                else:
                                                    length = 0.0
                                                    print(f"Fish {int(tracker_id)}: Unreliable depth data")
                                                
                                                #----- Update Database and Distance/Status Tracking ---
                                                n = 4
                                                update_db = False
                                                if tracker_id not in last_db_update or \
                                                (frame_count - last_db_update.get(tracker_id, 0)) > n:  # Every n frames
                                                    update_db = True
                                                    last_db_update[tracker_id] = frame_count 
                                                    current_position_3d = tracker.calculate_3d_centroid(head_coord, tail_coord)
                                                    if tracker_id in last_known_positions_3d:  # If fish has previous position
                                                        last_position_3d = last_known_positions_3d[tracker_id]  # Get last position
                                                        if last_position_3d != current_position_3d:
                                                            dist_m = length_estimate(current_position_3d, last_position_3d) / 1000.0  # Distance in meters
                                                        # TIME-BASED STATUS LOGIC
                                                        if dist_m >= MINIMAL_MOTION_THRESHOLD:  # If moved enough
                                                            current_total_distance_m += dist_m  # Add to total distance
                                                            last_active_time[tracker_id] = current_time  # Update last active time
                                                    
                                                    last_known_positions_3d[tracker_id] = current_position_3d  # Update last position
                                                    distance_traveled[tracker_id] = current_total_distance_m  # Update total distance
                                                    
                                                    # --- TIME-BASED STATUS DETECTION LOGIC ---
                                                    time_since_last_active = current_time - last_active_time.get(tracker_id, current_time)
                                                    
                                                    # Convert minutes to seconds
                                                    lethargy_timeout_seconds = LETHARGY_TIMEOUT_MINUTES * 60
                                                    dead_timeout_seconds = DEAD_TIMEOUT_MINUTES * 60
                                                    
                                                    # Determine status based on time
                                                    if time_since_last_active >= dead_timeout_seconds:
                                                        status = 'dead'
                                                    elif time_since_last_active >= lethargy_timeout_seconds:
                                                        status = 'lethargic'
                                                    else:
                                                        status = 'active'
                                                    
                                                    # Prepare location data for database
                                                    location_data_m = None  # Default location
                                                    if current_position_3d:  # If position available
                                                        location_data_m = {
                                                            "x": current_position_3d[0],
                                                            "y": current_position_3d[1],
                                                            "z": current_position_3d[2]
                                                        }
                                                if update_db:
                                                    DB.save_data_to_db(  # Save/update fish data in DB
                                                        fish_id=tracker_id,
                                                        size=length,
                                                        distance_traveled_m=current_total_distance_m,
                                                        is_active=True,
                                                        location_m=location_data_m,
                                                        status=status
                                                    )
                                
                                for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                                    x_min, y_min, x_max, y_max = xyxy    
                                    # --- Frame Annotation ---
                                    box_color = (0, 255, 0)  # Default box color (green)
                                    text_color = (0, 255, 0)  # Default text color (green)
                                    if status == 'dead':  # If dead
                                        box_color = (0, 0, 255)  # Red box
                                        text_color = (0, 0, 255)  # Red text
                                    elif status == 'lethargic':  # If lethargic
                                        box_color = (0, 255, 255)  # Yellow box
                                        text_color = (0, 255, 255)  # Yellow text
                                    
                                    # Draw Box & ID
                                    # Format time info for display
                                    time_since_last_active = current_time - last_active_time.get(tracker_id, current_time)
                                    label_text = f"ID {int(tracker_id)} | {confidence:.2f} | {status} | Dist: {dist_m:.2f} m | Len: {length:.1f} mm | Inactive: {time_since_last_active/60:.1f}min"
                                    cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 2)
                                    cv2.putText(color_image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                                    
                                # --- Handle Inactive Fish ---
                                ids_to_remove = [  # List of fish IDs to remove
                                    track_id for track_id, last_frame in last_seen_frame.items()
                                    if frame_count - last_frame > FRAME_TIMEOUT  # If not seen for too long
                                ]
                                for track_id in ids_to_remove:  # Loop through each inactive fish
                                    DB.save_data_to_db(  # Mark fish as inactive in DB
                                        fish_id=track_id,
                                        size=length,
                                        distance_traveled_m= DB.get_last_known_distance(track_id),
                                        is_active=False,
                                        location_m=None,
                                        status='inactive'
                                    )
                                    last_seen_frame.pop(track_id, None)  # Remove from last seen dict
                                    last_known_positions_3d.pop(track_id, None)  # Remove from positions dict
                                    distance_traveled.pop(track_id, None)  # Remove from distance dict
                                    last_active_time.pop(track_id, None)  # Remove from active time dict
                                    first_seen_time.pop(track_id, None)  # Remove from first seen time dict
                    except Exception as e:
                        print(f"Tracking error: {e}")
                
                # --- End frame monitoring and get stats ---
                frame_time = monitor.end_frame()
                stats = monitor.get_stats()

                # --- Update Display with fish counts ---
                y_offset = 110
                line_height = 30
                cv2.putText(color_image, f'Live Count: {live_fish_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)  # Show live count
                cv2.putText(color_image, f'Total Count: {total_unique_fish}', (10, 70 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)  # Show total count
                
                # --- Display Monitoring Information ---
                
                # cv2.putText(color_image, f'FPS: {stats["fps"]}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # cv2.putText(color_image, f'Host Frame Time: {frame_time:.1f}ms', (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                # cv2.putText(color_image, f'Host CPU: {stats["current_cpu"]:.1f}%', (10, y_offset + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                # cv2.putText(color_image, f'Frame #{frame_count}', (10, y_offset + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 2)

                # --- Print periodic stats to console ---
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"\n--- Monitoring Stats (Frame {frame_count}) ---")
                    print(f"FPS: {stats['fps']}")
                    print(f"Host Frame Time: {frame_time:.1f}ms (Avg: {stats.get('avg_frame_time', 0):.1f}ms)")
                    print(f"Host CPU: {stats['current_cpu']:.1f}% (Avg: {stats.get('avg_cpu', 0):.1f}%)")
                    print(f"Live Fish: {live_fish_count}")
                    print(f"Total Unique Fish: {total_unique_fish}")
                    print("-" * 40)

                # Write frame to video if recording
                if recording_started:
                    print("add frame to video")
                    video_recorder.add_frame(color_image)
                
                if no_cv2 is False:
                    # show video
                    with frame_lock:
                        cv2.imshow("key point ye", color_image)
            if no_cv2 is False:   
                key = cv2.waitKey(1)
                if key == ord("x"):
                    print("Got x key from keyboard!")
                    break
                if key == ord("r"):  # Toggle recording with 'r' key
                    if recording_started:
                        print("Stopping recording...")
                        video_recorder.stop_recording()
                        recording_started = False
                    else:
                        print("Starting recording...")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_recorder.start_recording(w, h, output_path=f"fish_tracking_{timestamp}.mp4", fps=fps_limit)
                        recording_started = True

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
            DB.close()
        except Exception as e :
            print(f"Error closing database: {e}")

        # Stop monitor and print final stats
        monitor.stop()
        print("\n--- Final Statistics ---")
        final_stats = monitor.get_stats()
        print(f"Total Frames Processed: {frame_count}")
        if frame_count > 0:
            print(f"Average FPS: {final_stats.get('fps', 0):.1f}")
            print(f"Average host Frame Time: {final_stats.get('avg_frame_time', 0):.1f}ms")
            print(f"Average host CPU Usage: {final_stats.get('avg_cpu', 0):.1f}%")
        print("Monitor stopped.")
        
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
    ap.add_argument('--yes-visualizer', action='store_true', help='Enable RemoteConnection visualizer')
    ap.add_argument("--webSocketPort", type=int, default=8765)
    ap.add_argument("--httpPort", type=int, default=8080)
    ap.add_argument('--no-cv2', action='store_true', help='Enable cv2 imshow visualizer')
    args = ap.parse_args()
    main(yes_visualizer=args.yes_visualizer, httpPort=args.httpPort, webSocketPort=args.webSocketPort,no_cv2=args.no_cv2)
