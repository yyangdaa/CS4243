import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Find interference lines as mask 
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 5]))

    # Find coloured regions as mask
    col = cv2.bitwise_not(cv2.inRange(hsv, np.array([0, 0, 250]), np.array([180, 50, 255])))
    col = cv2.bitwise_and(col, cv2.bitwise_not(mask))

    # Find intersection of regions
    inter = cv2.dilate(col, np.ones((3, 3), np.uint8), iterations=1)
    inter = cv2.bitwise_and(inter, mask)
    non_inter = cv2.bitwise_and(mask, cv2.bitwise_not(inter))

    # Remove non-intersection sections
    img_new = img.copy()
    img_new[non_inter > 0] = 255

    # Create median filter
    b, g, r = cv2.split(img_new)
    b_med, g_med, r_med = [cv2.medianBlur(c, 5) for c in cv2.split(img_new)]

    # Apply median filter on interference line pixels only
    b_new = np.where(inter == 0, b, b_med)
    g_new = np.where(inter == 0, g, g_med)
    r_new = np.where(inter == 0, r, r_med)

    # Convert to gray
    img_new = cv2.merge([b_new, g_new, r_new])
    result = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

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
