

# NeuroSight - Brain Tumor Classification System

## Overview
NeuroSight is a web-based application that uses deep learning to classify brain tumors from MRI scans. The system can identify three types of brain tumors:
- Glioma
- Meningioma
- Pituitary Tumor

## Features
- User-friendly web interface
- Real-time image processing
- High-accuracy tumor classification
- Confidence rate display for each tumor type
- Responsive design that works on both desktop and mobile devices

## Technical Stack
- **Frontend**: HTML, TailwindCSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: PIL, NumPy

## Project Structure

├── server.py           # Main Flask application
├── utils.py           # Utility functions for ML model and image processing
├── static/
│   ├── img/          # Static images including background
│   ├── upload/       # Uploaded brain scans
│   └── fonts/        # Custom fonts
└── templates/
    ├── index.html    # Home page
    ├── upload.html   # Image upload page
    └── results.html  # Results display page


## Installation

1. Clone the repository:

git clone https://github.com/yourusername/neurosight.git
cd neurosight


2. Install required packages:

pip install flask tensorflow pillow numpy werkzeug


3. Download the pre-trained model:
Place `brain_tumor_classifier.keras` in the root directory.

## Usage

1. Start the server:

python server.py


2. Open a web browser and navigate to:

http://localhost:5000


3. Click "Try It Now!" and upload a brain MRI scan
4. View the classification results and confidence rates

## Model Information
The system uses a Convolutional Neural Network (CNN) trained on brain MRI scans. The model:
- Accepts 512x512 RGB images
- Provides classification probabilities for three tumor types
- Automatically preprocesses and resizes uploaded images

## API Endpoints

- `GET /` - Home page
- `GET /upload` - Upload page
- `POST /predict` - Process uploaded image and return predictions
- `GET /results/<filename>` - Display results page

## Error Handling
The application includes comprehensive error handling for:
- Invalid file uploads
- Missing files
- Server processing errors
- Model prediction errors

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.