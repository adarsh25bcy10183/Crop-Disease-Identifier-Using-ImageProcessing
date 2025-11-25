# Crop-Disease-Identifier-Using-ImageProcessing
Course: Fundamentals in AI and ML  Student Name: ADARSH KUMAR RAIWAL Registration Number: 25BCY10183
This project is a Python-based console application designed to help farmers and agricultural workers quickly identify common plant leaf diseases by analyzing leaf images. It utilizes fundamental computer vision techniques (OpenCV) and structured algorithms to classify diseases without requiring complex, resource-intensive deep learning models.

✨ Features and Capabilities
Image Analysis: Reads and processes leaf images in common formats (e.g., JPEG, PNG).

Feature Extraction: Extracts image features using HSV Color Histograms.

Algorithmic Classification: Classifies the disease by comparing the input histogram against reference samples using the Bhattacharyya distance.

Dataset-Based: Uses a file-based dataset stored in folders (healthy, rust, blight) that can be easily expanded.


Remedy Suggestion: Provides a basic suggested action or remedy based on the identified disease.

Lightweight & Offline: The solution works without internet access and avoids complex database or GUI requirements.


⚙️ Technical Implementation
The system is built with a modular design, applying several core Python and Computer Vision concepts.

1. Core Concepts Used

Concept	Application in Project	Report Source
OpenCV (cv2)	
Used for reading image files, converting the color space (BGR to HSV ), and calculating the color histograms.

Functions (Modularity)	
Code is broken into modular functions (feature_extract, compare, identify) to improve reusability and follow the top-down problem-solving approach.


Control Structures	
For loops iterate through category folders and compare the input image against every sample. If statements manage decisions for minimum score selection and disease determination.


Lists	
Python lists are used to store dataset paths, category names, feature scores, and comparison results.

File Handling (Image)	
OpenCV is used to read image files, which serve as the persistent input data for the system.


2. Classification Algorithm (Histogram Comparison)

The system identifies diseases using the following steps:

Read & Convert: The input leaf image is read and converted to the HSV (Hue, Saturation, Value) color space.

Extract Features: A 2D HSV Histogram is calculated for the image.

Compare: The input histogram is compared against the pre-calculated histograms of all reference samples in the dataset folders (healthy, rust, blight).

Determine Match: The Bhattacharyya distance is computed for each comparison. The category that yields the minimum distance (highest similarity) is selected as the predicted disease.

🚀 Usage Instructions
Prerequisites

Python 3.x

OpenCV (cv2) library:


