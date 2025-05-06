# RiceLeaf-Disease-Detection
Rice plant disease classification using CNN and pretrained models
🌾 Rice Leaf Disease Detection Using Deep Learning
📌 Project Overview
This project focuses on classifying three major rice plant leaf diseases using Convolutional Neural Networks (CNN) and pretrained models. Early detection of plant diseases helps improve crop yield and reduce loss.

📁 Dataset Overview
Source: Custom dataset from DataMites

Total Images: 120 JPG images

Classes:

🍃 Bacterial Leaf Blight

🍂 Brown Spot

🍁 Leaf Smut

Images are equally distributed across the 3 classes.

🎯 Objectives
Build a CNN model to classify rice leaf diseases.

Apply transfer learning using pretrained models (VGG16, ResNet50, MobileNetV2, EfficientNet).

Compare model performance and select the best.

Improve prediction accuracy through data preprocessing and augmentation.

🔧 Tools & Technologies
Language: Python

Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn

Techniques:

Convolutional Neural Networks (CNN)

Transfer Learning

Image Preprocessing

Data Augmentation

🧪 Workflow
Data Loading & Exploration

Image loading using ImageDataGenerator

Display sample images for each class

Preprocessing

Resizing images to 224x224

Normalizing pixel values

Augmenting training data (rotation, flip, zoom)

Model Building

Custom CNN model

Pretrained models: VGG16, ResNet50, MobileNetV2, EfficientNetB0

Model Evaluation

Accuracy, Precision, Recall, F1 Score

Confusion Matrix & Classification Report

Visualization of training vs. validation loss/accuracy

Model Comparison

Compared all models to select the best performer

MobileNetV2 achieved highest accuracy and F1 Score (≈ 88%)

📊 Results
Model	Accuracy	F1 Score
CNN (Custom)	72%	0.75
MobileNetV2	88%	0.88
VGG16	84%	0.85
ResNet50	85%	0.86
EfficientNet	87%	0.87

✅ Key Insights
MobileNetV2 was lightweight and gave the best balance between performance and training time.

Data augmentation helped improve model generalization.

Confusion matrix showed clear separation between classes with MobileNetV2.

🧠 Skills Applied
Deep Learning (CNN)

Transfer Learning

Model Evaluation Techniques

Visualization & Reporting

Image Classification

📂 Folder Structure
mathematica
Copy
Edit
RiceLeaf-Disease-Detection/
├── rice_leaf_dataset/
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
├── RiceLeaf_Classification.ipynb
├── model_comparison.png
├── README.md
└── requirements.txt
🌱 Future Work
Deploy the best model using Streamlit or Flask

Expand dataset with more samples for better generalization

Integrate mobile-based prediction app for farmers

🙌 Acknowledgment
Thanks to DataMites™ and mentor Rajesh Amalraj for guidance throughout the project.

