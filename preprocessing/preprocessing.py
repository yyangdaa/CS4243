import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path, 0)

    # Obtain line mask of interference lines
    _, line_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)

    # Remove interference lines
    result = img.copy()
    result[line_mask == 255] = 255

    # Sharpen image before binarization
    blurred = cv2.GaussianBlur(result, (0, 0), 3)
    sharpened = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)

    # Get final binary image
    _, result_binary = cv2.threshold(sharpened, 250, 255, cv2.THRESH_BINARY_INV)

    return result_binary

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of PNG files only
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    print(f"Found {len(image_files)} PNG files to process")
    
    # Process each image with progress bar
    for filename in tqdm(image_files, desc="Processing images"):
        # Construct full input path
        input_path = os.path.join(input_folder, filename)
        
        # Construct output path with same filename
        output_path = os.path.join(output_folder, filename)
        
        # Process the image
        processed_img = preprocess_image(input_path)
        
        # Save the processed image
        cv2.imwrite(output_path, processed_img)
    
    print(f"Successfully processed {len(image_files)} images. Results saved to {output_folder}")

input_folder = "original_images"
output_folder = "preprocessed_images"
process_folder(input_folder, output_folder)
