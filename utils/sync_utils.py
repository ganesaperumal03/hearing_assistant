"""
Synchronization and matching utilities for Audio/Video streams in real-time captions.
"""

import time
from collections import deque
import threading
from typing import List, Tuple, Any, Optional


class SlidingWindowBuffer:
    """
    A thread-safe buffer that retains elements within a sliding window of time.
    Useful for aligning video frames and audio chunks chronologically.
    """
    def __init__(self, window_duration_sec: float = 5.0):
        self.window_duration = window_duration_sec
        self.queue = deque()
        self.lock = threading.Lock()

    def append(self, data: Any, timestamp: Optional[float] = None) -> None:
        """Appends data with a corresponding timestamp (defaults to current time)."""
        t = timestamp or time.time()
        with self.lock:
            self.queue.append((t, data))
            # Sort queue to handle potential out-of-order delivery
            self.queue = deque(sorted(self.queue, key=lambda x: x[0]))
            self._prune(t)

    def _prune(self, current_time: float) -> None:
        """Removes data outside the sliding window duration."""
        cutoff = current_time - self.window_duration
        while self.queue and self.queue[0][0] < cutoff:
            self.queue.popleft()

    def get_all(self) -> List[Tuple[float, Any]]:
        """Returns all elements in the buffer."""
        with self.lock:
            return list(self.queue)

    def get_within_range(self, start_time: float, end_time: float) -> List[Any]:
        """Retrieves items falling strictly within a specific time range."""
        with self.lock:
            return [data for t, data in self.queue if start_time <= t <= end_time]

    def get_closest_match(self, target_time: float, max_delta_sec: float = 0.05) -> Optional[Any]:
        """
        Finds the closest element in the buffer to the given target_time,
        provided it falls within the max_delta_sec tolerance.
        """
        with self.lock:
            if not self.queue:
                return None
            best_match = None
            min_diff = max_delta_sec
            for t, data in self.queue:
                diff = abs(t - target_time)
                if diff <= min_diff:
                    min_diff = diff
                    best_match = data
            return best_match

    def clear(self) -> None:
        """Clears the buffer."""
        with self.lock:
            self.queue.clear()


def match_audio_video_frames(
    audio_events: List[Tuple[float, Any]],
    video_events: List[Tuple[float, Any]],
    max_delta_sec: float = 0.05
) -> List[Tuple[Any, Any]]:
    """
    Matches audio events (e.g. voice activity segments) and video frames 
    that fall within a maximum time difference threshold (default: 50ms).
    """
    matched = []
    video_sorted = sorted(video_events, key=lambda x: x[0])
    
    for audio_t, audio_data in audio_events:
        # Find closest video frame in time
        best_match = None
        min_diff = max_delta_sec
        
        for video_t, video_data in video_sorted:
            diff = abs(audio_t - video_t)
            if diff < min_diff:
                min_diff = diff
                best_match = video_data
                
        if best_match is not None:
            matched.append((audio_data, best_match))
            
    return matched
