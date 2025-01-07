# **AI-Powered Image and Video Colorization Using Deep Learning**  

## **Overview**

This project explores the application of deep learning in transforming grayscale images and videos into realistic, colorized versions. By utilizing pre-trained convolutional neural networks (CNNs) and advanced image-processing techniques, we demonstrate how AI can effectively bridge the gap between monochrome and vibrant visuals. The project also provides an interactive Streamlit-based web interface, allowing users to upload, process, and customize both images and videos.  

---  

## **Purpose of the Project**  

The goal of this project is to develop an efficient and user-friendly tool for restoring and enhancing grayscale media. It focuses on applying state-of-the-art deep learning models to automate the colorization process for both static images and dynamic video content, showcasing the versatility of CNNs in multimedia processing.  

---

## **Technologies Used**  

1. **Python**: The core programming language used for processing and integration.  
2. **OpenCV**: For image and video processing, including frame extraction and color space transformations.  
3. **NumPy**: For handling numerical computations.  
4. **Streamlit**: To create an intuitive web-based interface for user interaction.  
5. **Deep Learning Model Components**:  
   - **colorization\_release\_v2.caffemodel**: Pre-trained model for colorization.  
   - **models\_colorization\_deploy\_v2.prototxt**: Defines the architecture of the deployed model.  
   - **pts\_in\_hull.npy**: Contains cluster centers for "ab" channels in the LAB color space.  

---  

## **How Deep Learning is Used**  

The project employs CNNs trained on the ImageNet dataset to predict the chrominance components ("ab") of the LAB color space for grayscale media. By combining these predictions with the luminance channel ("L"), we achieve realistic colorization.  

### **Image Colorization**  
1. **Feature Extraction**: Extract spatial features from grayscale images using CNN layers.  
2. **Color Prediction**: Predict chrominance values using a fully connected layer.  
3. **Post-Processing**: Combine predictions with the grayscale channel and apply LAB-to-RGB conversion to generate the final image.  

### **Video Colorization**  
1. **Frame-by-Frame Processing**: Extract individual frames from videos and colorize them using the image model.  
2. **Reconstruction**: Merge the colorized frames back into a continuous video with smooth transitions.  

---  

## **Features**  

### **Image Colorization**  
1. **Interactive Web Interface**: Upload grayscale images and instantly view results.  
2. **Customizable Outputs**: Adjust color intensity, saturation, and hue for tailored results.  
3. **Download Options**: Save the colorized images locally.  

### **Video Colorization**  
1. **Automated Frame Processing**: Apply the image colorization model to every frame.  
2. **Batch Processing**: Handle high-resolution videos with efficiency.  
3. **Seamless Reconstruction**: Generate high-quality videos with consistent coloring across frames.  

### **Scalability**  
- Efficient handling of high-resolution images and lengthy videos.  

---  

## **Advantages**  

1. **Ease of Use**: A user-friendly interface via Streamlit.  
2. **Pre-Trained Model**: Eliminates the need for training, reducing computational overhead.  
3. **High-Quality Outputs**: Produces realistic and vibrant colorizations.  
4. **Modular Design**: Adaptable for additional functionalities, such as video editing or real-time processing.  

---  

## **Disadvantages**  

1. **Inconsistent Results**: Challenges in handling complex scenes or outlier images/videos.  
2. **Dependency on Pre-Trained Models**: Limited generalization to datasets outside the training domain.  
3. **Computational Cost**: Processing high-resolution videos can be time-intensive.  

---  

## **Why These Components Are Used**  

### **colorization\_release\_v2.caffemodel**  
- **Purpose**: Stores the pre-trained neural network weights.  
- **Why It’s Used**: Reduces training time by using pre-learned features.  

### **models\_colorization\_deploy\_v2.prototxt**  
- **Purpose**: Specifies the network structure for inference.  
- **Why It’s Used**: Ensures accurate deployment and compatibility.  

### **pts\_in\_hull.npy**  
- **Purpose**: Contains cluster centers for chrominance values ("ab" components) in LAB space.  
- **Why It’s Used**: Accelerates inference and improves color accuracy.  

---  

## **Installation and Requirements**  

### **Packages Needed**  

- Python 3.8 or later  
- OpenCV  
- NumPy  
- Streamlit  
- FFmpeg (for video processing)  

Install the required packages using:  

```bash  
pip install opencv-python-headless numpy streamlit ffmpeg-python  
```
or

```bash
pip install -r requirements.txt
```

---  

## **How to Run**  

1. Clone the repository:  

   ```bash  
   git clone <repository_url>  
   cd <repository_directory>  
   ```  

2. Place the model files in the `models/` directory:  
   - `colorization_release_v2.caffemodel`(Download link: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)
   - `models_colorization_deploy_v2.prototxt`  
   - `pts_in_hull.npy`  

3. Run the Streamlit app:  

   ```bash  
   python -m streamlit run app.py
   ```
   or

   ```bash  
   streamlit run app.py
   ```  

5. Open the provided URL in your browser, upload an image or video, and process it!  

---
# **Channels**

## **What are Channels in an Image?**  
- Channels are like layers of information in an image.  
- Think of an image as a **sandwich**, where each channel is one **layer** of that sandwich. Together, they create the final image.  

---

## **What is the L and ab?**  
In this project, we are working with the **LAB Color Space**. This is another way to represent colors in an image, splitting it into **three channels**:  

### **1. L (Lightness):**  
- Represents the **brightness** of the image (how light or dark it is).  
- Focuses on shades of gray and doesn't involve color.  
- This is why we use the **L channel** from grayscale images.  

### **2. a (Green-Red):**  
- Captures the **color information** on a scale from green to red.  
  - Greenish pixels = lower values.  
  - Reddish pixels = higher values.  

### **3. b (Blue-Yellow):**  
- Captures the **color information** on a scale from blue to yellow.  
  - Bluish pixels = lower values.  
  - Yellowish pixels = higher values.  

Together, the **L**, **a**, and **b** channels combine to give a vibrant colored image.  

---

## **How Does It Work in This Project?**  

### **1. Input:**  
- Start with a **grayscale image**, which only contains the **L channel** (lightness/brightness).  

### **2. Prediction:**  
- The deep learning model predicts the **a** and **b** channels (color details) based on its training and patterns it has learned.  

### **3. Reconstruction:**  
- Combine the predicted **a** and **b** channels with the original **L channel** to create a fully colorized image in the LAB space.  

### **4. Conversion to RGB:**  
- Convert the LAB image into **RGB (Red-Green-Blue)** format, so it can be displayed on screens and devices.

---

## **Simple Analogy:**  
Imagine you're watching a **black-and-white movie**:  
- The **L channel** is like the brightness levels of the movie.  
- The **a channel** adds shades of **red and green**.  
- The **b channel** adds shades of **blue and yellow**.  

When mixed together, it turns into a **full-color movie!** 🎥✨  

---
# **Magic of converting black-and-white (grayscale) media to color✨**

### 1. **Starting Point: The Black-and-White Image**
- A grayscale image only contains information about **brightness**. 
- Think of it as a map showing **light (white)** and **dark (black)** areas, but it doesn’t know what color each part should be.

---

### 2. **LAB Color Space: Adding Color**
- To colorize the image, we use the **LAB color space**:
  - **L (Lightness)**: Holds the brightness information (what you already have in grayscale).
  - **A**: Holds the green-to-red color values.
  - **B**: Holds the blue-to-yellow color values.

So, the challenge is to **predict the "A" and "B" channels** and combine them with the "L" channel.

---

### 3. **How the Model Predicts Colors**
1. **Training the Model**:
   - The model has seen **millions of real, full-color images** during training.
   - It learns patterns like:
     - Grass is usually green.
     - The sky is often blue.
     - Faces have skin tones.
   
2. **Input the Grayscale Image**:
   - You feed the model just the "L" channel (brightness info).

3. **Prediction**:
   - The model guesses the "A" and "B" channels (color info) for every pixel based on the patterns it learned.

---

### 4. **Combining Channels**
- After the model predicts the "A" and "B" channels:
  - The "L" channel from the grayscale image is combined with the predicted "A" and "B" channels.
  - This creates a full-color image in the LAB color space.
- Finally, the LAB image is converted to the **RGB color space** (what we see on screens).

---

### 5. **The Result: A Colorized Image**
- The grayscale image now has realistic colors!
- It might not be **perfect** (e.g., the model might make a shirt blue when it was actually red), but it’s remarkably close for a machine!

---

### Simplified Analogy 🎨
Imagine coloring a grayscale coloring book:
1. The black-and-white lines tell you where to color (the "L" channel).
2. You use your brain (the model) to guess what colors go where based on your knowledge (training).
3. You fill in the colors (predict "A" and "B"), and voilà—a colorful picture! 😊
---

## **How Video Colorization Works**

The video colorization process builds on the image colorization methodology but involves additional steps to handle multiple frames:  

1. **Input:**  
   - The input video is split into individual **frames** (images).  
   - Each frame is treated as a grayscale image, containing only the **L channel**.  

2. **Frame-by-Frame Colorization:**  
   - Each frame is processed through the deep learning model to predict the **a** and **b** channels.  
   - The colorized frames are then reconstructed by combining the predicted channels with the original **L channel**.  

3. **Temporal Consistency:**  
   - To avoid flickering or abrupt color changes, temporal consistency techniques are applied:  
     - **Optical flow** tracks motion between frames, helping maintain smooth transitions.  
     - **Regularization** ensures similar regions across frames have consistent colors.  

4. **Reconstruction:**  
   - The colorized frames are stitched back together to create the final video.  

5. **Output:**  
   - The LAB color space video is converted to RGB and saved as a colorized video file.  

---

### **Simple Analogy for Video Colorization**  
- Imagine a flipbook with grayscale drawings (frames).  
- You color each page using the same process as image colorization.  
- To ensure smooth transitions, you compare consecutive pages to maintain consistent colors.  
- When you flip through the book, the entire sequence looks like a fully colored animation!
---
## **Contributors**  

This project was developed as part of a college project by a team of 4 students exploring deep learning techniques for image and video processing.  

1. **Hanish B**  
2. **Mohamadu Riyas S**  
3. **Surya**  
4. **Logdharshan**  

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
