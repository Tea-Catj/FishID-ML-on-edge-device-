import numpy as np
import supervision as sv

class TrackerHandler:
    def __init__(self, track_thresh: float = 0.3, track_buffer: int = 50):
        # 1. Initialize ByteTrack
        # Note: Using the new parameter names compatible with recent supervision versions
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )

    def _nn_results_to_detections(self, raw_detections, frame_shape) -> sv.Detections:
        """
        Converts raw NN detection data from DepthAI into supervision.Detections format.
        """
        if not raw_detections:
            return sv.Detections.empty()

        # Lists to collect data from all detections
        xyxy_list = []
        confidence_list = []
        class_id_list = []
        keypoints_list = []

        h, w, _ = frame_shape

        for d in raw_detections:
            # --- 1. Extract Bounding Box ---
            # Accessing center.x/y and size.width/height from the nested rotated_rect object
            if hasattr(d, 'rotated_rect'):
                rect = d.rotated_rect
                x_center = rect.center.x
                y_center = rect.center.y
                width    = rect.size.width
                height   = rect.size.height

                # Calculate corners (normalized)
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                # Scale to pixels immediately
                xyxy_list.append([
                    int(x_min * w), 
                    int(y_min * h), 
                    int(x_max * w), 
                    int(y_max * h)
                ])
            else:
                continue

            # --- 2. Extract Confidence and Label ---
            confidence_list.append(d.confidence)
            class_id_list.append(d.label)

            # --- 3. Extract Keypoints ---
            if hasattr(d, 'keypoints') and len(d.keypoints) > 0:
                current_kps = []
                for kp in d.keypoints:
                    # Robust check for Keypoint object attributes vs flat values
                    if hasattr(kp, 'x') and hasattr(kp, 'y'):
                        current_kps.append([kp.x, kp.y])
                    else:
                        current_kps.append([kp, kp]) 
                keypoints_list.append(current_kps)
            else:
                keypoints_list.append([])

        # If no valid detections were processed
        if not xyxy_list:
            return sv.Detections.empty()

        # Convert lists to NumPy arrays
        xyxy = np.array(xyxy_list)
        scores = np.array(confidence_list)
        class_ids = np.array(class_id_list)

        # Process Keypoints into (N, K, 2)
        keypoints_pixel = None
        if len(keypoints_list) > 0 and len(keypoints_list[0]) > 0:
            kp_array_norm = np.array(keypoints_list, dtype=np.float32)
            scale_vec = np.array([w, h])
            # Scale and cast to int
            keypoints_pixel = (kp_array_norm * scale_vec).astype(int)

        # --- FIX: Pass keypoints via 'data' argument ---
        data_dict = {}
        if keypoints_pixel is not None:
            data_dict['keypoints'] = keypoints_pixel

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=scores,
            class_id=class_ids,
            data=data_dict  # Passed here instead of 'keypoints='
        )
    
        return detections

    def get_tracked_results(self, raw_detections, frame_shape) -> sv.Detections:
        """
        Converts raw data, updates the tracker, and returns tracked detections.
        """
        # 1. Convert raw NN data to supervision.Detections
        detections = self._nn_results_to_detections(raw_detections, frame_shape)
        
        # 2. Update the tracker
        tracked_detections = self.byte_tracker.update_with_detections(detections)

        # --- FIX: Re-attach keypoints for Main.py ---
        # Main.py expects 'tracked_detections.keypoints', but ByteTrack stores custom fields in '.data'
        if tracked_detections.data and 'keypoints' in tracked_detections.data:
            tracked_detections.keypoints = tracked_detections.data['keypoints']
        else:
            # Ensure the attribute exists to prevent AttributeErrors in Main.py
            tracked_detections.keypoints = None
        

        return tracked_detections