import cv2
import numpy as np
import os

# --- Configuration ---
# Define categories used in the project report
CATEGORIES = ["healthy", "rust", "blight"]
DATASET_DIR = "dataset" 
IMG_SIZE = (256, 256) # Standard size for processing

# --- Feature Extraction ---

def feature_extract(img):
    """
    Converts image to HSV color space, resizes it, and computes the 2D 
    (Hue and Saturation) color histogram.
    """
    if img is None:
        return None

    # Resize for standardization
    img = cv2.resize(img, IMG_SIZE)
    
    # Convert to HSV (separates color from light intensity)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate Histogram (Hue and Saturation)
    hist = cv2.calcHist(
        [hsv_img],              # Source image
        [0, 1],                 # Channels: 0=Hue, 1=Saturation
        None,                   # No mask
        [18, 18],               # Histogram size (18 bins for each)
        [0, 180, 0, 256]        # Range for H (0-180) and S (0-256)
    )
    
    # Normalize the histogram (crucial for accurate comparison)
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX).flatten()
    return hist

# --- Dataset Loading ---

def load_reference_histograms(base_dir=DATASET_DIR):
    """
    Loads and extracts histograms for all reference images in the dataset folders.
    """
    reference_data = {}
    print("⏳ Loading and processing dataset...")

    for category in CATEGORIES:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            print(f"⚠️ Warning: Dataset path not found: {category_path}")
            continue
            
        reference_data[category] = []
        
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    hist = feature_extract(img)
                    reference_data[category].append(hist)
        
        print(f"✅ Loaded {len(reference_data[category])} samples for {category}")

    return reference_data

# --- Classification (Comparison) ---

def compare_histograms(input_hist, reference_histograms):
    """
    Compares the input histogram against reference histograms using the 
    Bhattacharyya distance metric.
    """
    best_match_score = float('inf')  # Lower score is better
    predicted_disease = "Unknown"

    for category, hists in reference_histograms.items():
        if not hists:
            continue
            
        total_distance = 0
        for ref_hist in hists:
            # cv2.HISTCMP_BHATTACHARYYA is the distance metric specified in the report.
            distance = cv2.compareHist(input_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
            total_distance += distance

        # Calculate the average distance for this entire category
        avg_distance = total_distance / len(hists)
        
        # Check for the minimum distance (best match)
        if avg_distance < best_match_score:
            best_match_score = avg_distance
            predicted_disease = category
            
    return predicted_disease, best_match_score