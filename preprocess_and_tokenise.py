import cv2
import numpy as np
import os
import sys

MIN_AREA = 10
MAX_AREA = 5000
KERNEL_SIZE = (3, 3)
OUTPUT_SIZE = (28, 28)

def load_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        print("Error: Could not load image", filepath)
        sys.exit(1)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.medianBlur(equalized, 3)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

def find_valid_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA <= area <= MAX_AREA:
            valid_contours.append(cnt)
    return valid_contours

def sort_contours_left_to_right(contours):
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    sorted_indices = np.argsort([box[0] for box in bounding_boxes])
    sorted_contours = [contours[i] for i in sorted_indices]
    return sorted_contours

def extract_and_normalize_characters(binary, contours):
    tokens = []
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, w, h))
        char_img = binary[y:y+h, x:x+w]
        norm_img = cv2.resize(char_img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        tokens.append(norm_img)
    return tokens, bounding_boxes

def annotate_image(image, bounding_boxes):
    annotated = image.copy()
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return annotated

def process_single_image(filepath):
    image = load_image(filepath)
    binary = preprocess_image(image)
    contours = find_valid_contours(binary)
    sorted_contours = sort_contours_left_to_right(contours)
    tokens, bounding_boxes = extract_and_normalize_characters(binary, sorted_contours)
    annotated = annotate_image(image, bounding_boxes)
    return tokens, bounding_boxes, annotated

def main():
    if len(sys.argv) < 3:
        print("Usage: python segmentation_tokenization.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    supported_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
             if os.path.isfile(os.path.join(input_folder, f)) and os.path.splitext(f)[1].lower() in supported_ext]
    for file in files:
        print("Processing", file)
        tokens, bounding_boxes, annotated = process_single_image(file)
        base_name = os.path.splitext(os.path.basename(file))[0]
        annotated_path = os.path.join(output_folder, base_name + "_annotated.png")
        cv2.imwrite(annotated_path, annotated)
        for idx, token in enumerate(tokens):
            token_path = os.path.join(output_folder, base_name + "_char_" + str(idx) + ".png")
            cv2.imwrite(token_path, token)
        print("Saved results for", base_name)

if __name__ == "__main__":
    main()
