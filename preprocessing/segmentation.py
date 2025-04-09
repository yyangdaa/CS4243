import cv2
import numpy as np
from tqdm import tqdm
import os

input_folder = "preprocessed_images"
output_folder = "character_dataset"

indexDict = {}

def segment_image(filename):
    input_path = os.path.join(input_folder, filename)
    img = cv2.imread(input_path, 0)
    num_labels, labels = cv2.connectedComponents(img)

    string_label = os.path.splitext(filename)[0]
    expected_char_count = len(string_label)

    # Close gaps untile expected number of components is reached
    k = 3
    while num_labels - 1 > expected_char_count:
        kernel = np.ones((k, k), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        num_labels, labels = cv2.connectedComponents(closed)
        k += 1

    # If lesser components than expected, overlapping case 
    if num_labels - 1 < expected_char_count:
        return 0
    
    # Create subdirectories for each character in this image
    for char in string_label:
        char_dir = os.path.join(output_folder, char)
        os.makedirs(char_dir, exist_ok=True)
    
    components = []

    for label in range(1, num_labels):  # Skip background (label 0)
        # Create mask for this component
        mask = np.uint8(labels == label) * 255
        
        # Find bounding box coordinates
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extract the character from the original image
        char_img = img[y:y+h, x:x+w]

        components.append([x, char_img])
    
    components.sort(key=lambda k: k[0])
    
    for i, component in enumerate(components):
        char_label = string_label[i]

        idx = indexDict.get(char_label, 0)
        indexDict[char_label] = indexDict.get(char_label, 0) + 1

        output_path = os.path.join(output_folder, char_label, f"{char_label}_{idx:04d}.png")

        cv2.imwrite(output_path, component[1])

    return 1

def process_folder():
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of PNG files only
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    print(f"Found {len(image_files)} PNG files to process")
    
    success = 0

    # Process each image with progress bar
    for filename in tqdm(image_files, desc="Processing images"):        
        success += segment_image(filename)
    
    print(f"Successfully processed {success}/{len(image_files)} images. Results saved to {output_folder}")

process_folder()