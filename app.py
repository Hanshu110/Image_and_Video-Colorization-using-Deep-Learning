import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
from io import BytesIO
from video_colorizer import colorize_video  # Import the video colorizer function
from enhancements import apply_style_transfer # Import enhancement functions

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

def colorizer(img, net, upsample_method=cv2.INTER_CUBIC):
    # Ensure input image has 3 channels (convert grayscale to RGB)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Store original dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Convert to grayscale, then back to RGB for consistency
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Preserve original L channel for later merging
    original_L = cv2.split(lab)[0].copy()

    # Resize for network input
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Use higher quality upsampling method
    ab = cv2.resize(ab, (orig_w, orig_h), interpolation=upsample_method)

    # Use original L channel for better detail preservation
    colorized = np.concatenate((original_L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized

def color_correction(image, reference_image=None):
    """
    Perform color correction based on reference image or general corrections 
    to reduce yellowish cast and improve color accuracy
    """
    if reference_image is not None:
        try:
            # Ensure reference_image has the same dimensions as image
            reference_image = cv2.resize(reference_image, (image.shape[1], image.shape[0]))
            
            # Color transfer from reference image
            src_mean, src_std = cv2.meanStdDev(cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB))
            dst_mean, dst_std = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
            
            # Convert to LAB for better color transfer
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            
            # Avoid division by zero
            if dst_std[1][0] == 0: dst_std[1][0] = 1
            if dst_std[2][0] == 0: dst_std[2][0] = 1
            
            # Adjust A and B channels (color)
            a = ((a - dst_mean[1][0]) * (src_std[1][0]/dst_std[1][0])) + src_mean[1][0]
            b = ((b - dst_mean[2][0]) * (src_std[2][0]/dst_std[2][0])) + src_mean[2][0]
            
            # Make sure all channels have the same dimensions
            if not (l.shape == a.shape == b.shape):
                raise ValueError("Channel dimensions don't match")
                
            # Merge channels
            lab = cv2.merge([l, a, b])
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return corrected
        except Exception as e:
            st.warning(f"Color correction failed: {e}. Using auto-correction instead.")
            # Fall through to auto-correction
    
    # General correction to reduce yellowish cast
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Slightly reduce yellow hues
    yellow_mask = ((h >= 20) & (h <= 35))
    h[yellow_mask] = np.clip(h[yellow_mask] - 5, 0, 179)
    s[yellow_mask] = np.clip(s[yellow_mask] * 0.85, 0, 255)  # Reduce saturation of yellows
    
    # Merge channels
    corrected = cv2.merge([h, s, v])
    corrected = cv2.cvtColor(corrected, cv2.COLOR_HSV2RGB)
    
    return corrected

def enhance_details(image, amount=1.0):
    """
    Enhance details in the image by applying unsharp mask
    """
    # Convert to LAB color space to preserve colors while sharpening
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply unsharp mask to L channel only
    gaussian = cv2.GaussianBlur(l, (0, 0), 3)
    unsharp_mask = cv2.addWeighted(l, 1.5, gaussian, -0.5, 0)
    
    # Apply amount control
    enhanced_l = cv2.addWeighted(l, 1 - amount, unsharp_mask, amount, 0)
    
    # Recombine channels
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

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

def try_super_resolution(image):
    try:
        # Check if we have a super-resolution model
        if os.path.exists("models/ESPCN_x2.pb"):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel("models/ESPCN_x2.pb")
            sr.setModel("espcn", 2)  # 2x upscaling
            
            # Apply SR
            upscaled = sr.upsample(image)
            
            # Resize back to original dimensions
            upscaled = cv2.resize(upscaled, (image.shape[1], image.shape[0]))
            return upscaled
        else:
            return image
    except Exception as e:
        st.warning(f"Super resolution failed: {e}")
        return image

st.title("Advanced Colorize Your Black-and-White Media with AI ✨")

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
                # Model selection
                model_choice = st.sidebar.selectbox(
                    "Colorization Model",
                    ["Standard", "Artistic", "Realistic (if available)"]
                )

                if model_choice == "Standard":
                    prototxt = "models/models_colorization_deploy_v2.prototxt"
                    model = "models/colorization_release_v2.caffemodel"
                    points = "models/pts_in_hull.npy"
                elif model_choice == "Artistic":
                    prototxt = "models/models_colorization_deploy_v2.prototxt"
                    model = "models/colorization_release_v2_norebal.caffemodel"
                    points = "models/pts_in_hull.npy"
    
    # Check if artistic model exists, use standard if not
                    if not os.path.exists(model):
                        st.warning("Artistic model not found, using standard model instead.")
                        model = "models/colorization_release_v2.caffemodel"
                else:  # Realistic option
                    prototxt = "models/models_colorization_deploy_v2.prototxt"
                    model = "models/colorization_release_v2.caffemodel"  # Fallback to standard
                    points = "models/pts_in_hull.npy"
                    if model_choice == "Realistic (if available)":
                        st.warning("Realistic model not found, using standard model instead")
                net = load_model(prototxt, model, points)
                if net:
                    # Choose interpolation method
                    interpolation_method = st.sidebar.selectbox(
                        "Interpolation Quality",
                        ["Cubic (Better Quality)", "Linear (Faster)"]
                    )
                    upsample_method = cv2.INTER_CUBIC if interpolation_method == "Cubic (Better Quality)" else cv2.INTER_LINEAR

                    # Basic colorization with selected method
                    colorized_img = colorizer(img, net, upsample_method=upsample_method)
                    
                    # Add color correction options
                    correction_method = st.sidebar.radio(
                        "Color Correction",
                        ["None", "Auto-correct", "Upload reference image"]
                    )
                    
                    if correction_method == "Auto-correct":
                        colorized_img = color_correction(colorized_img)
                    elif correction_method == "Upload reference image":
                        ref_file = st.sidebar.file_uploader("Upload color reference image", type=["jpg", "png"])
                        if ref_file:
                            ref_image = np.array(Image.open(ref_file).convert("RGB"))
                            ref_image = resize_image(ref_image)
                            colorized_img = color_correction(colorized_img, ref_image)
                    
                    # Add detail enhancement slider
                    detail_amount = st.sidebar.slider("Detail Enhancement", 0.0, 1.0, 0.5)
                    if detail_amount > 0:
                        colorized_img = enhance_details(colorized_img, detail_amount)
                
                    # Intensity adjustment slider
                    intensity = st.sidebar.slider("Adjust Color Intensity", 0.5, 2.0, 1.0)
                    colorized_img = adjust_intensity(colorized_img, intensity)

                    # Hue and saturation sliders
                    hue = st.sidebar.slider("Adjust Hue", -90, 90, 0)
                    saturation = st.sidebar.slider("Adjust Saturation", 0.5, 2.0, 1.0)
                    colorized_img = adjust_hue_saturation(colorized_img, hue, saturation)
                    
                    # Apply style transfer
                    style_type = st.sidebar.selectbox("Apply Style Transfer", ["None", "Vintage", "Sepia", "HDR"])
                    if style_type != "None":
                        colorized_img = apply_style_transfer(colorized_img, style_type)

                    # Option for super-resolution (if model is available)
                    use_sr = st.sidebar.checkbox("Try Super Resolution (if available)")
                    if use_sr:
                        colorized_img = try_super_resolution(colorized_img)

                    # Region of Interest (ROI) selection
                    use_roi = st.sidebar.checkbox("Apply colorization to specific region")
                    if use_roi:
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
                    color_pil.save(buf, format="JPEG", quality=95)  # Higher quality save
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="Download Colorized Image",
                        data=byte_im,
                        file_name="colorized_image.jpg",
                        mime="image/jpeg"
                    )
                    # Trigger balloons on successful colorization
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

                # Video colorization settings
                # Model selection
                model_choice = st.sidebar.selectbox(
                    "Colorization Model",
                    ["Standard", "Artistic", "Realistic (if available)"]
                )
                
                # Color correction
                correction_method = st.sidebar.radio(
                    "Color Correction for Video",
                    ["None", "Auto-correct"]
                )
                
                # Detail enhancement
                detail_amount = st.sidebar.slider("Video Detail Enhancement", 0.0, 1.0, 0.5)
                
                # Output path for the colorized video
                output_video_path = "output_colorized_video.mp4"

                # Call the video colorization function with parameters
                colorize_video(
                    temp_input_path, 
                    output_video_path, 
                    model_type=model_choice.lower(),
                    apply_correction=(correction_method=="Auto-correct"),
                    detail_enhancement=detail_amount
                )
                
                st.balloons()
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
