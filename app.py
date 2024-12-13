import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
from io import BytesIO

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

st.title("Colorize Your Black-and-White Images ✨")

file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "png"])

if file:
    try:
        image = Image.open(file)
        img = np.array(image)
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

                st.text("Your colorized image")
                st.image(colorized_img, use_container_width=True)

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
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.text("You haven't uploaded an image file❌")