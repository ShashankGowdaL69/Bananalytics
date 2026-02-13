# Bananalytics: Predictive and Prescriptive Analytics for Banana Ripeness

Bananalytics is an end-to-end Machine Learning expert system designed to go beyond standard image classification. Powered by a fine-tuned PyTorch ResNet-18 architecture, it analyzes the visual characteristics of a banana to predict its current ripeness stage, estimate remaining shelf life, and prescribe actionable use cases based on food science principles.

This project demonstrates a complete AI pipeline, from handling real-world data augmentations to deploying a fully interactive web application.

## Features

### Intelligent Ripeness Classification
Accurately classifies inputs into Unripe, Ripe, Overripe, or Rotten using a deep convolutional neural network.

### Shelf-Life Prediction
Estimates the remaining days before spoilage based on model confidence scores.

### Prescriptive Recommendations
Suggests the optimal culinary use cases and storage methods depending on the current ripeness stage.

### Spoilage Risk Assessment
Categorizes the risk level to assist in preventing food waste.

### Continuous Ripeness Score
Calculates a weighted 0-100 index for precise, continuous tracking rather than simple discrete labels.

## Image Upload Guidelines & Dataset Context
This model was trained on the Banana Ripeness Classification Dataset (Kaggle), which consists of over 11,000 images of real-world bananas. Because the model learned from authentic photography rather than digital graphics, your input images must match this context for accurate predictions.

### For best results, please upload images that are:
* Real-world photos: Captured with a smartphone or standard camera.
* Naturally lit: Standard indoor or daylight conditions with natural shadows.
* Contextualized: Resting on a real surface (e.g., a table, cutting board, or counter).
* Textured: Showing natural blemishes, ridges, and stems.

### Avoid uploading:
* Stock graphics: Perfectly smooth, glossy, or digitally altered images.
* Isolated backgrounds: Images with a pure white, artificially removed background.
* Unnatural lighting: Harsh, blinding studio flashes that wash out the fruit's true color.

## Sample Images for Testing
To help you get started immediately, we have included a `sample_images/` folder in this repository. These images perfectly match the environmental context and lighting conditions the model was trained on. Feel free to use these to test the intelligence engine!

## Tech Stack
* Deep Learning Framework: PyTorch
* Model Architecture: ResNet-18 (Transfer Learning with frozen base layers for computational efficiency)
* Frontend / UI: Streamlit
* Data Processing: NumPy, Pillow (PIL), Torchvision

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/ShashankGowdaL69/Bananalytics.git](https://github.com/ShashankGowdaL69/Bananalytics.git)
   cd Bananalytics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Intelligence Engine:
   ```bash
   streamlit run app.py
   ```
