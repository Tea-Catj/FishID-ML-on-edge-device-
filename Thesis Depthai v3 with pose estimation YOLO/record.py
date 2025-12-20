
import cv2
import threading
import queue
from datetime import datetime
import os

class ThreadedVideoRecorder:
    def __init__(self, max_queue_size=30):
        self.recording = False
        self.writer = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        
    def start_recording(self, frame_width, frame_height, output_path="output.mp4", fps=30):
        """Start recording with given parameters"""
        if self.recording:
            self.stop_recording()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Start worker thread
        self.recording = True
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._write_frames, daemon=True)
        self.worker_thread.start()
        
        print(f"Started recording: {output_path}")
        return output_path
    
    def add_frame(self, frame):
        """Add frame to write queue"""
        if self.recording:
            # Non-blocking put, drop frames if queue is full
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Drop frame if queue is full
    
    def _write_frames(self):
        """Worker thread function to write frames"""
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
    
    def stop_recording(self):
        """Stop recording and clean up"""
        self.recording = False
        self.stop_event.set()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                break
        
        print("Stopped recording")
    
    def is_recording(self):
        return self.recording
    
    def cleanup(self):
        self.stop_recording()