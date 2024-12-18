import cv2
import numpy as np

# Paths to pre-trained model files
PROTOTXT_PATH = "models/models_colorization_deploy_v2.prototxt"
CAFFEMODEL_PATH = "models/colorization_release_v2.caffemodel"
POINTS_PATH = "models/pts_in_hull.npy"

# Load the pre-trained colorization model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

# Load cluster centers
pts = np.load(POINTS_PATH).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

def colorize_frame(frame):
    """Colorizes a single frame using the pre-trained model."""
    h, w = frame.shape[:2]
    scaled_frame = frame.astype(np.float32) / 255.0
    lab_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2Lab)
    l_channel = lab_frame[:, :, 0]
    
    # Prepare input for the network
    l_channel_resized = cv2.resize(l_channel, (224, 224))  # Model expects 224x224 input
    l_channel_resized = l_channel_resized - 50  # Normalize
    
    net.setInput(cv2.dnn.blobFromImage(l_channel_resized))
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))  # Get predicted AB channels
    ab_channels_resized = cv2.resize(ab_channels, (w, h))

    # Combine with original L channel
    lab_colorized = np.concatenate((l_channel[:, :, np.newaxis], ab_channels_resized), axis=2)
    bgr_colorized = cv2.cvtColor(lab_colorized, cv2.COLOR_Lab2BGR)
    bgr_colorized = np.clip(bgr_colorized, 0, 1)

    return (bgr_colorized * 255).astype(np.uint8)

def colorize_video(input_video_path, output_video_path):
    """Colorizes a video frame-by-frame and saves the output."""
    cap = cv2.VideoCapture(input_video_path)
    writer = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        colorized_frame = colorize_frame(frame)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        writer.write(colorized_frame)

    cap.release()
    if writer:
        writer.release()

    print("Video colorization completed!")

if __name__ == "__main__":
    input_video_path = "data/bw_video.mp4"  # Path to input black-and-white video
    output_video_path = "data/colorized_video.mp4"  # Path to save colorized video

    colorize_video(input_video_path, output_video_path)
