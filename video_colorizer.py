import cv2
import numpy as np
from typing import Optional

class VideoColorizer:
    def __init__(self):
        self.models = {
            'standard': {
                'prototxt': "models/models_colorization_deploy_v2.prototxt",
                'model': "models/colorization_release_v2.caffemodel",
                'points': "models/pts_in_hull.npy"
            },
            'artistic': {
                'prototxt': "models/models_colorization_deploy_v2.prototxt",
                'model': "models/colorization_release_v2_norebal.caffemodel",
                'points': "models/pts_in_hull.npy"
            }
        }
        self.current_net = None
        
    def load_model(self, model_type: str = 'standard'):
        """Load the specified colorization model."""
        model_config = self.models.get(model_type.lower(), self.models['standard'])
        
        try:
            net = cv2.dnn.readNetFromCaffe(model_config['prototxt'], model_config['model'])
            pts = np.load(model_config['points'])
            
            # Set up network points
            pts = pts.transpose().reshape(2, 313, 1, 1)
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            self.current_net = net
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def enhance_details(self, frame: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Enhance details in the frame using unsharp mask."""
        if amount <= 0:
            return frame
            
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        gaussian = cv2.GaussianBlur(l, (0, 0), 3)
        unsharp_mask = cv2.addWeighted(l, 1.5, gaussian, -0.5, 0)
        enhanced_l = cv2.addWeighted(l, 1 - amount, unsharp_mask, amount, 0)
        
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply automatic color correction to reduce color casting."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Reduce yellow cast
        yellow_mask = ((h >= 20) & (h <= 35))
        h[yellow_mask] = np.clip(h[yellow_mask] - 5, 0, 179)
        s[yellow_mask] = np.clip(s[yellow_mask] * 0.85, 0, 255)
        
        corrected = cv2.merge([h, s, v])
        return cv2.cvtColor(corrected, cv2.COLOR_HSV2BGR)

    def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Colorize a single frame."""
        if self.current_net is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        h, w = frame.shape[:2]
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare input
        scaled = frame.astype("float32") / 255.0
        lab = cv2.cvtColor(cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        # Colorize
        self.current_net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.current_net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize back to original dimensions
        ab = cv2.resize(ab, (w, h))
        
        # Merge with original L channel
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # Convert to BGR
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized * 255, 0, 255).astype("uint8")
        
        return colorized

def colorize_video(
    input_path: str,
    output_path: str,
    model_type: str = 'standard',
    apply_correction: bool = False,
    detail_enhancement: float = 0.0
) -> None:
    """
    Colorize a video with specified parameters.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        model_type: Type of colorization model ('standard', 'artistic', 'realistic')
        apply_correction: Whether to apply color correction
        detail_enhancement: Amount of detail enhancement (0.0 to 1.0)
    """
    try:
        # Initialize colorizer
        colorizer = VideoColorizer()
        
        # Load appropriate model
        if not colorizer.load_model(model_type):
            raise ValueError(f"Failed to load {model_type} model")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Failed to open input video")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Colorize frame
            colorized = colorizer.colorize_frame(frame)
            
            # Apply optional enhancements
            if apply_correction:
                colorized = colorizer.apply_color_correction(colorized)
            
            if detail_enhancement > 0:
                colorized = colorizer.enhance_details(colorized, detail_enhancement)
            
            writer.write(colorized)
        
        # Clean up
        cap.release()
        writer.release()
        
    except Exception as e:
        raise Exception(f"Video colorization failed: {str(e)}")
