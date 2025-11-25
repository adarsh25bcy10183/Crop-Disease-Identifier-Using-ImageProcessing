import cv2
import os
from disease_identifier import (
    feature_extract, 
    load_reference_histograms, 
    compare_histograms,
    DATASET_DIR
)

def display_remedy(disease):
    """Provides a basic suggested remedy based on the identified disease."""
    print("\n--- 🛠️ Suggested Action ---")
    if disease == "healthy":
        print("🌱 The leaf appears healthy. Continue normal watering and maintenance.")
        print("Suggestion: Monitor regularly for new spots or changes.")
    elif disease == "rust":
        print("🔴 Rust Disease Detected! This is often caused by fungus.")
        print("Action: Remove infected leaves immediately. Apply a fungicidal spray containing copper or sulfur.")
    elif disease == "blight":
        print("⚫ Blight Disease Detected! This can be fungal or bacterial.")
        print("Action: Apply appropriate fungicide (if fungal) or bactericide. Improve air circulation and reduce overhead watering.")
    else:
        print("❓ Classification inconclusive or unknown. Check the image quality.")

def run_identifier():
    """Main function to run the crop disease identifier application."""
    print("=== Crop Disease Identifier using Image Processing ===")
    
    # 1. Load reference data once
    try:
        reference_histograms = load_reference_histograms()
        if not any(reference_histograms.values()):
            print(f"❌ ERROR: No reference images found in the '{DATASET_DIR}' folder. Cannot proceed.")
            print("Please ensure you have images in the 'dataset/healthy', 'dataset/rust', and 'dataset/blight' subfolders.")
            return
            
    except Exception as e:
        print(f"An error occurred during dataset loading: {e}")
        return

    while True:
        input_path = input("\nEnter path to the leaf image (or type 'exit' to quit): ").strip()
        
        if input_path.lower() == 'exit':
            break
            
        if not os.path.exists(input_path):
            print("❌ File not found. Please check the path and try again.")
            continue

        # 2. Load and extract features from the user's input image
        input_img = cv2.imread(input_path)
        if input_img is None:
            print("❌ Could not read the image. Ensure the file is a valid image format (jpg/png).")
            continue
            
        input_hist = feature_extract(input_img)
        
        # 3. Classify
        predicted_disease, score = compare_histograms(input_hist, reference_histograms)
        
        # 4. Display results
        print("\n--- 🔬 Analysis Result ---")
        print(f"Input Image: {os.path.basename(input_path)}")
        print(f"Predicted Disease: **{predicted_disease.upper()}**")
        print(f"Similarity Score (Bhattacharyya Distance): {score:.4f} (Lower is Better)")

        display_remedy(predicted_disease)
        print("-" * 35)

    print("\n👋 Application closed.")

if __name__ == "__main__":
    run_identifier()