import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

def medianBlur(img, ksize, mask, iterations=1):
    """
    Applies the median blur only to TRUE pixels within the image
    for the given number of iterations
    """
    b, g, r = cv2.split(img)

    for _ in range(iterations):
        b_med, g_med, r_med = [cv2.medianBlur(c, ksize) for c in (b, g, r)]
        b = np.where(mask, b_med, b)
        g = np.where(mask, g_med, g)
        r = np.where(mask, r_med, r)
    return cv2.merge([b, g, r])

def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path)

    # Find interference lines as mask
    black = np.full((1, 3), 0, dtype=np.uint8)
    line_mask = cv2.inRange(img, black, black)

    # Find character regions as mask
    white = np.full((1, 3), 255, dtype=np.uint8)
    back_mask = cv2.inRange(img, white, white)
    chr_mask = cv2.bitwise_and(cv2.bitwise_not(back_mask),
                               cv2.bitwise_not(line_mask))

    # Find intersection through morphological operations
    chr_dilate = cv2.dilate(chr_mask, np.ones((3, 3), np.uint8), iterations=1)
    line_chr = cv2.bitwise_and(chr_dilate, line_mask)
    line_not_chr = cv2.bitwise_and(line_mask,
                                   cv2.bitwise_not(chr_dilate))

    # Remove non-intersection regions
    img_new = img.copy()
    img_new[line_not_chr > 0] = 255

    # Apply median filter on interference line pixels only
    img_new = medianBlur(img_new, 3, line_chr == 255, iterations=5)

    # Convert to gray
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
