# ResNet Cancer Prediction Project

This project implements a ResNet-based model for multi-class cancer prediction, distinguishing between glioma, meningioma, no tumor, and pituitary tumor classes.

## Dataset
The dataset consists of 6,000 images, categorized into four classes:
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

### Data Preparation
The dataset is split into training and testing datasets:
- **Training Data:** Create a `Training` folder and add data for training the data.
- **Testing Data:** Create a `Testing` folder and add data for testing the data.

## Model Architecture

The project utilizes a **ResNet** architecture for image classification. Dropout is implemented to reduce overfitting.

- Input Shape: `(224, 224, 3)`
- Number of Classes: `4`

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
