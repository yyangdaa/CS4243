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

    # Obtain line mask of interference lines
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 5]))

    # Replace interference lines with median
    h, s, v = cv2.split(hsv)
    h_med, s_med, v_med = [cv2.medianBlur(c, 5) for c in cv2.split(hsv)]
    h_new = cv2.bitwise_and(h, h, mask=cv2.bitwise_not(mask)) + cv2.bitwise_and(h_med, h_med, mask=mask)
    s_new = cv2.bitwise_and(s, s, mask=cv2.bitwise_not(mask)) + cv2.bitwise_and(s_med, s_med, mask=mask)
    v_new = cv2.bitwise_and(v, v, mask=cv2.bitwise_not(mask)) + cv2.bitwise_and(v_med, v_med, mask=mask)

    # Convert to gray
    hsv_new = cv2.merge([h_new, s_new, v_new])
    img_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
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
