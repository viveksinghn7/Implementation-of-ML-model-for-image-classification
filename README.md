# Implementation of ML Model for Image Classification

This project demonstrates image classification using two different models: MobileNetV2 (pre-trained on ImageNet) and a custom CIFAR-10 model. The application is built using Streamlit, a framework for creating interactive web applications in Python. Users can upload images and receive predictions with confidence scores from either model. It features a sleek navigation bar for easy switching and real-time results, which is ideal for learning and practical use.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [MobileNetV2](#mobilenetv2)
  - [CIFAR-10](#cifar-10)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/viveksinghn7/Implementation-of-ML-model-for-image-classification.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload an image (JPG or PNG) and choose the model type (MobileNetV2 or CIFAR-10) from the sidebar.

4. The application will classify the uploaded image and display the predicted class along with the confidence score.

## Models

### MobileNetV2

MobileNetV2 is a pre-trained model on the ImageNet dataset. It is designed for efficient image classification on mobile and embedded devices.

### CIFAR-10

The CIFAR-10 model is a custom-trained model on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
