import time
import psutil
import collections
import threading

class SystemMonitor:
    def __init__(self, history_size=30):
        self.history_size = history_size
        self.frame_times = collections.deque(maxlen=history_size)
        self.cpu_history = collections.deque(maxlen=history_size)
        self.fps_history = collections.deque(maxlen=history_size)
        self.frame_count = 0
        self.total_frame_time = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # For threaded CPU monitoring
        self.cpu_usage = 0
        self.monitoring = True
        self.cpu_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        self.cpu_thread.start()

    def _monitor_cpu(self):
        """Background thread to monitor CPU usage"""
        while self.monitoring:
            self.cpu_usage = psutil.cpu_percent(interval=0.5)
            self.cpu_history.append(self.cpu_usage)
    
    def start_frame(self):
        """Call at the beginning of each frame"""
        self.frame_start_time = time.time()
        self.frame_count += 1
        self.fps_counter += 1
    
    def end_frame(self):
        """Call at the end of each frame"""
        frame_time = (time.time() - self.frame_start_time) * 1000
        self.frame_times.append(frame_time)
        
        # Update FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_history.append(self.current_fps)
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        return frame_time
    
    def get_stats(self):
        """Get current monitoring statistics"""
        stats = {
            'fps': self.current_fps,
            'frame_count': self.frame_count,
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0,
            'max_frame_time': max(self.frame_times) if self.frame_times else 0,
            'min_frame_time': min(self.frame_times) if self.frame_times else 0,
            'current_cpu': self.cpu_usage,
            'avg_cpu': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
        }
        return stats
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.cpu_thread.is_alive():
            self.cpu_thread.join(timeout=1.0)