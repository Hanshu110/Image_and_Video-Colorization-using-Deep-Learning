import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
from io import BytesIO
from video_colorizer import colorize_video  # Import the video colorizer function

def resize_image(img, max_dim=512):
    height, width = img.shape[:2]
    if max(height, width) > max_dim:
        
        scaling_factor = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))
    return img

def load_model(prototxt, model, points):
    if not os.path.exists(prototxt):
        st.error(f"Prototxt file not found: {prototxt}")
        return None
    if not os.path.exists(model):
        st.error(f"Caffe model file not found: {model}")
        return None
    if not os.path.exists(points):
        st.error(f"Points file not found: {points}")
        return None

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

def colorizer(img, net):
    # Ensure input image has 3 channels (convert grayscale to RGB)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale, then back to RGB for consistency
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized

def adjust_intensity(colorized_img, intensity=1.0):
    return np.clip(colorized_img * intensity, 0, 255).astype(np.uint8)

def adjust_hue_saturation(image, hue=0, saturation=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_color_to_roi(image, net, roi):
    (x, y, w, h) = roi
    roi_region = image[y:y+h, x:x+w]
    colorized_roi = colorizer(roi_region, net)
    image[y:y+h, x:x+w] = colorized_roi
    return image

def side_by_side_comparison(original, colorized):
    comparison = np.hstack((original, colorized))
    return comparison

st.title("Colorize Your Black-and-White Media with AI ✨")

option = st.sidebar.selectbox("Choose an option:", ["Image Colorizer", "Video Colorizer"])

if option == "Image Colorizer":
    file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "png"])

    if file:
        try:
            # Load image with PIL and ensure 3 channels
            image = Image.open(file).convert("RGB")
            img = np.array(image)

            # Resize the image
            img = resize_image(img)

            st.text("Your original image")
            st.image(image, use_container_width=True)

            with st.spinner("Colorizing... Please wait!"):
                net = load_model(
                    "models/models_colorization_deploy_v2.prototxt",
                    "models/colorization_release_v2.caffemodel",
                    "models/pts_in_hull.npy"
                )
                if net:
                    colorized_img = colorizer(img, net)

                    # Intensity adjustment slider
                    intensity = st.sidebar.slider("Adjust Color Intensity", 0.5, 2.0, 1.0)
                    colorized_img = adjust_intensity(colorized_img, intensity)

                    # Hue and saturation sliders
                    hue = st.sidebar.slider("Adjust Hue", -90, 90, 0)
                    saturation = st.sidebar.slider("Adjust Saturation", 0.5, 2.0, 1.0)
                    colorized_img = adjust_hue_saturation(colorized_img, hue, saturation)

                    # Region of Interest (ROI) selection
                    roi_x = st.sidebar.number_input("ROI X", 0, img.shape[1] - 1, 0)
                    roi_y = st.sidebar.number_input("ROI Y", 0, img.shape[0] - 1, 0)
                    roi_w_max = img.shape[1] - roi_x
                    roi_h_max = img.shape[0] - roi_y
                    roi_w = st.sidebar.number_input("ROI Width", 1, roi_w_max, roi_w_max)
                    roi_h = st.sidebar.number_input("ROI Height", 1, roi_h_max, roi_h_max)

                    if roi_w > 0 and roi_h > 0:
                        roi = (roi_x, roi_y, roi_w, roi_h)
                        colorized_img = apply_color_to_roi(colorized_img, net, roi)

                    # Display comparison
                    st.text("Comparison: Original vs. Colorized")
                    comparison = side_by_side_comparison(img, colorized_img)
                    st.image(comparison, use_container_width=True)

                    # Download option
                    color_pil = Image.fromarray(colorized_img)
                    buf = BytesIO()
                    color_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="Download Colorized Image",
                        data=byte_im,
                        file_name="colorized_image.jpg",
                        mime="image/jpeg"
                    )
                    # **Trigger balloons on successful colorization** date 23.01.2025
                    st.balloons()
                    st.success("Image colorization completed successfully! 🎉")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.text("You haven't uploaded an image file❌")

elif option == "Video Colorizer":
    video_file = st.sidebar.file_uploader("Upload a black-and-white video", type=["mp4", "avi"])

    if video_file:
        try:
            with st.spinner("Colorizing video... Please wait!"):
                # Save the uploaded file to a temporary location
                temp_input_path = "temp_input_video.mp4"
                with open(temp_input_path, "wb") as temp_file:
                    temp_file.write(video_file.read())

                # Output path for the colorized video
                output_video_path = "output_colorized_video.mp4"

                # Call the video colorization function
                colorize_video(temp_input_path, output_video_path)
                st.balloons() # Update V-101 date 23.01.2025
                st.success("Video colorization completed!")

                # Show a download button
                with open(output_video_path, "rb") as video:
                    st.download_button(
                        label="Download Colorized Video",
                        data=video,
                        file_name="colorized_video.mp4",
                        mime="video/mp4"
                    )

                # Clean up the temporary file
                os.remove(temp_input_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.text("You haven't uploaded a video file❌")
