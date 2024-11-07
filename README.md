# 🎯 NeuroSight - Brain Tumor Classification System

![NeuroSight Banner](static/img/banner.png)

## 📖 Overview
**NeuroSight** is a web-based application that uses deep learning to classify brain tumors from MRI scans. It can identify three types of brain tumors:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

## ✨ Features
- **User-friendly web interface**
- **Real-time image processing**
- **High-accuracy tumor classification**
- **Confidence rate display for each tumor type**
- **Responsive design** for both desktop and mobile devices

## 🛠️ Technical Stack
| **Component**     | **Technologies**        |
|-------------------|-------------------------|
| **Frontend**      | HTML, TailwindCSS, JavaScript |
| **Backend**       | Flask (Python)          |
| **Machine Learning** | TensorFlow/Keras      |
| **Image Processing** | PIL, NumPy          |

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/neurosight.git
    cd neurosight
    ```

2. **Install required packages:**
    ```bash
    pip install flask tensorflow pillow numpy werkzeug
    ```

3. **Download the pre-trained model:**
   Place `brain_tumor_classifier.keras` in the root directory.

## 💡 Usage

1. **Start the server:**
    ```bash
    python server.py
    ```

2. **Open a web browser and navigate to:**
    ```
    http://localhost:5000
    ```

3. **Upload an MRI scan** on the home page and view classification results.

## 🧠 Model Information
This system leverages a Convolutional Neural Network (CNN) trained on brain MRI scans. Key details:
- **Image Format**: 512x512 RGB
- **Output**: Classification probabilities for tumor types
- **Automatic Preprocessing**: Resize and preprocess uploaded images

## 🌐 API Endpoints
- **`GET /`** - Home page
- **`GET /upload`** - Upload page
- **`POST /predict`** - Processes uploaded images and returns predictions
- **`GET /results/<filename>`** - Displays the results page

## ⚠️ Error Handling
The application includes error handling for:
- Invalid file uploads
- Missing files
- Server processing errors
- Model prediction errors

## 🤝 Contributing
Contributions are welcome! Feel free to fork the project and submit a pull request. 

---