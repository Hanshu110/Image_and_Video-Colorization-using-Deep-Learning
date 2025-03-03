# enhancements.py
import cv2
import numpy as np

def apply_style_transfer(img, style_type):
    if style_type == "Vintage":
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    elif style_type == "Sepia":
        img = cv2.transform(img, np.array([[0.272, 0.534, 0.131], 
                                           [0.349, 0.686, 0.168], 
                                           [0.393, 0.769, 0.189]]))
    elif style_type == "HDR":
        img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return np.clip(img, 0, 255).astype(np.uint8)
