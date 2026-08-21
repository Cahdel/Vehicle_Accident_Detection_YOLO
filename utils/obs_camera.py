"""
Utility module for streaming to OBS Virtual Camera
Handles initialization, frame writing, and cleanup of OBS virtual camera stream
"""

import cv2
import numpy as np
import pyvirtualcam
import threading
from typing import Optional, Tuple

class OBSVirtualCamera:
    """
    Class to handle streaming to OBS Virtual Camera
    """
    
    def __init__(self, width: int = 640, height: int = 360, fps: int = 25):
        """
        Initialize OBS Virtual Camera stream
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_streaming = False
        self.frame_lock = threading.Lock()
        
    def start(self) -> bool:
        """
        Start OBS Virtual Camera stream
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        try:
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            self.is_streaming = True
            print(f"✅ OBS Virtual Camera started: {self.width}x{self.height} @ {self.fps} FPS")
            return True
        except Exception as e:
            print(f"❌ Failed to start OBS Virtual Camera: {str(e)}")
            self.is_streaming = False
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to OBS Virtual Camera
        
        Args:
            frame: Image frame (BGR format from OpenCV)
            
        Returns:
            bool: True if successfully written, False otherwise
        """
        if not self.is_streaming or self.camera is None:
            return False
        
        try:
            with self.frame_lock:
                # Ensure frame is the correct size
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert BGR to RGB for pyvirtualcam
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.camera.send(frame_rgb)
                return True
        except Exception as e:
            print(f"❌ Error writing frame to OBS Virtual Camera: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop OBS Virtual Camera stream
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        try:
            if self.camera is not None:
                self.camera.close()
                self.is_streaming = False
                print("✅ OBS Virtual Camera stopped")
                return True
        except Exception as e:
            print(f"❌ Error stopping OBS Virtual Camera: {str(e)}")
        
        return False
    
    def is_active(self) -> bool:
        """Check if streaming is active"""
        return self.is_streaming and self.camera is not None


# Global instance for easy access
_obs_camera: Optional[OBSVirtualCamera] = None


def get_obs_camera() -> Optional[OBSVirtualCamera]:
    """Get the global OBS camera instance"""
    global _obs_camera
    return _obs_camera


def init_obs_camera(width: int = 640, height: int = 360, fps: int = 25) -> OBSVirtualCamera:
    """
    Initialize the global OBS camera instance
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        
    Returns:
        OBSVirtualCamera: The initialized camera instance
    """
    global _obs_camera
    if _obs_camera is None:
        _obs_camera = OBSVirtualCamera(width, height, fps)
    return _obs_camera


def send_frame_to_obs(frame: np.ndarray) -> bool:
    """
    Send frame to OBS Virtual Camera
    
    Args:
        frame: Image frame (BGR format from OpenCV)
        
    Returns:
        bool: True if successfully sent, False otherwise
    """
    camera = get_obs_camera()
    if camera is None:
        return False
    return camera.write_frame(frame)


def cleanup_obs_camera():
    """Clean up OBS Virtual Camera resources"""
    global _obs_camera
    if _obs_camera is not None:
        _obs_camera.stop()
        _obs_camera = None
