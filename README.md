# 🏞️ Image Classifier

A deep learning web application built with **Streamlit** and **TensorFlow** that classifies uploaded images into one of six natural scene categories: buildings, forest, glacier, mountain, sea, or street.

## 📸 App Link

[image-classifier-app](https://image-classifier-checkpoint.streamlit.app/)

## ✨ Features

- Upload images in popular formats (PNG, JPG, JPEG, JFIF, WEBP)
- Preprocessing: resizing to 150x150 pixels, normalization, and RGB conversion
- Real-time prediction using a pre-trained CNN model
- Displays predicted class and confidence score with a progress bar
- Responsive layout with sidebar instructions

## 🧠 Model

The classifier uses a **Convolutional Neural Network (CNN)** trained on the [Intel Image Classification dataset](https://www.kaggle.com/puneet6060/intel-image-classification) (or similar). The model is saved as `CNN_IC.keras` and expects input images of size 150x150x3.

### Classes:
| Index | Class       |
|-------|-------------|
| 0     | buildings   |
| 1     | forest      |
| 2     | glacier     |
| 3     | mountain    |
| 4     | sea         |
| 5     | street      |

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- Pillow (PIL)
- NumPy


