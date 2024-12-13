# Image Colorization using Deep Learning

## **Overview**

This project demonstrates a deep learning approach to colorizing black-and-white images. Using pre-trained models and advanced deep learning techniques, grayscale images are transformed into realistic colorized versions. The project provides a Streamlit-based web interface to upload and process images interactively.

---

## **Purpose of the Project**

The primary goal of this project is to explore how deep learning can be applied to image processing tasks, specifically converting grayscale images to color. It also aims to create a user-friendly application that demonstrates the practical applications of convolutional neural networks (CNNs) in image processing.

---

## **Technologies Used**

1. **Python**: The core programming language used for processing and integration.
2. **OpenCV**: For image processing and color space transformations.
3. **NumPy**: For handling numerical computations.
4. **Streamlit**: To create an intuitive web-based interface.
5. **Deep Learning Model Components**:
   - **colorization\_release\_v2.caffemodel**: Pre-trained model for colorization.
   - **models\_colorization\_deploy\_v2.prototxt**: Defines the architecture of the deployed model.
   - **pts\_in\_hull.npy**: Contains cluster centers for ab channels in the LAB color space, critical for colorization.

---

## **How Deep Learning is Used**

This project leverages a convolutional neural network (CNN) pre-trained on the ImageNet dataset. The CNN predicts the "ab" components of the LAB color space for each pixel in a grayscale image (the "L" component). By combining the predicted "ab" components with the original "L" channel, the image is converted to a full-color version. Key components include:

1. **Feature Extraction**: The model uses CNN layers to extract spatial features from the grayscale image.
2. **Color Prediction**: A fully connected layer predicts chrominance values based on extracted features.
3. **Post-Processing**: Predicted values are combined with the original grayscale image, and LAB-to-RGB conversion is applied to generate the final colorized image.

---

## **Features**

1. **Interactive Web Interface**: Allows users to upload images and view results instantly.
2. **Pre-Trained Model**: Utilizes a model trained on a large dataset for accurate colorization.
3. **Download Option**: Users can download the colorized image for further use.
4. **Scalability**: Efficient processing even for high-resolution images.

---

## **Advantages**

1. **Ease of Use**: The Streamlit interface ensures user-friendliness.
2. **Pre-Trained Model**: Eliminates the need for training, reducing computational overhead.
3. **High-Quality Outputs**: The model produces realistic colorization for most images.
4. **Modular Design**: Easy to extend and adapt for other image processing tasks.

---

## **Disadvantages**

1. **Inconsistent Results**: May fail to produce accurate colors for complex scenes or objects.
2. **Dependency on Pre-Trained Model**: Cannot generalize well to images outside the training dataset.
3. **Computational Cost**: Processing high-resolution images can be time-consuming.

---

## **Why These Components Are Used**

### **colorization\_release\_v2.caffemodel**

- **Purpose**: Contains weights and biases for the neural network pre-trained on the ImageNet dataset.
- **Why It’s Used**: Reduces the need for training a model from scratch, saving time and resources.

### **models\_colorization\_deploy\_v2.prototxt**

- **Purpose**: Specifies the network architecture for deploying the colorization model.
- **Why It’s Used**: Ensures the model is structured correctly for inference.

### **pts\_in\_hull.npy**

- **Purpose**: Provides cluster centers for chrominance values ("ab" components) in the LAB color space.
- **Why It’s Used**: Acts as a lookup table to reduce computational complexity and improve inference speed.

---

## **Installation and Requirements**

### **Packages Needed**

- Python 3.8 or later
- OpenCV
- NumPy
- Streamlit

Install the required packages using:

```bash
pip install opencv-python-headless numpy streamlit
```
or
```
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
   streamlit run app.py
   ```
   or
   ```
   python -m streamlit run app.py
   ```

5. Open the provided URL in your browser and upload a grayscale image to see the results.

---

## **Contributors**

This project was developed as part of a college project by a team of 4 students exploring deep learning techniques for image processing.

1. HANISH B
2. MOHAMADU RIYAS
3. SURYA
4. LOGDHARSHAN
