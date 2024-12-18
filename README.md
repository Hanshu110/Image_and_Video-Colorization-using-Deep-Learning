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
   - `colorization_release_v2.caffemodel`  
   - `models_colorization_deploy_v2.prototxt`  
   - `pts_in_hull.npy`  

3. Run the Streamlit app:  

   ```bash  
   python -m streamlit run app.py
   ```  

5. Open the provided URL in your browser, upload an image or video, and process it!  

---  

## **Contributors**  

This project was developed as part of a college project by a team of 4 students exploring deep learning techniques for image and video processing.  

1. **Hanish B**  
2. **Mohamadu Riyas S**  
3. **Surya**  
4. **Logdharshan**  
