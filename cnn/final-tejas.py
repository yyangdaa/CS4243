import cv2
import numpy as np
import os
import sys
import random

# Global parameters.
MIN_AREA = 100  # Minimum area (in pixels) to consider a region as a valid character.
WIDE_THRESHOLD_MULTIPLIER = 120  # If region width > 1.2x the median width, try to split it by color.
K_CLUSTERS = 2  # Number of clusters for k-means clustering in color splitting.
COLOR_SPLIT_MIN_RATIO = 0.005  # Minimum fraction of pixels (0.5%) for a cluster to be accepted.


def load_image(filename):
    image = cv2.imread(filename)
    if image is None:
        print("Error: Could not load image:", filename)
        sys.exit(1)
    return image


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def equalize_and_blur(gray_image):
    equalized = cv2.equalizeHist(gray_image)
    blurred = cv2.medianBlur(equalized, 3)
    return blurred


def binarize_image(blurred_image):
    binary = cv2.adaptiveThreshold(
        blurred_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed


def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_bounding_boxes(contours):
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= MIN_AREA:
            boxes.append((x, y, w, h))
    return boxes


def sort_boxes_left_to_right(boxes):
    return sorted(boxes, key=lambda b: b[0])


def median_width(boxes):
    widths = [box[2] for box in boxes]
    if not widths:
        return 0
    return np.median(widths)


def color_split_region(region_color):
    """Split a wide region by performing k-means clustering in HSV color space."""
    hsv_region = cv2.cvtColor(region_color, cv2.COLOR_BGR2HSV)
    pixel_values = hsv_region.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    labels_image = labels.reshape((region_color.shape[0], region_color.shape[1]))
    total_pixels = region_color.shape[0] * region_color.shape[1]
    segments = []
    for k in range(K_CLUSTERS):
        mask = np.uint8((labels_image == k) * 255)
        if cv2.countNonZero(mask) < COLOR_SPLIT_MIN_RATIO * total_pixels:
            continue
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest_contour = max(cnts, key=cv2.contourArea)
            sx, sy, sw, sh = cv2.boundingRect(largest_contour)
            if sw * sh >= MIN_AREA:
                segment = region_color[sy:sy + sh, sx:sx + sw]
                segments.append((sx, sy, sw, sh, segment))
    return segments


def process_image(image_path, output_folder):
    """
    Process a single image: load, preprocess, segment characters, and save them.
    For wide regions (merged characters), attempt to split by color.
    """
    image = load_image(image_path)
    gray = convert_to_grayscale(image)
    blurred = equalize_and_blur(gray)
    binary = binarize_image(blurred)
    contours = find_contours(binary)
    boxes = get_bounding_boxes(contours)
    sorted_boxes = sort_boxes_left_to_right(boxes)
    med_width = median_width(sorted_boxes)

    final_segments = []
    for box in sorted_boxes:
        x, y, w, h = box
        region_color = image[y:y + h, x:x + w]
        # If the region is wider than expected, try to split it.
        if w > (WIDE_THRESHOLD_MULTIPLIER / 100.0) * med_width:
            splits = color_split_region(region_color)
            if len(splits) > 1:
                for (sx, sy, sw, sh, seg_img) in splits:
                    final_segments.append((x + sx, y + sy, sw, sh, seg_img))
            else:
                final_segments.append((x, y, w, h, region_color))
        else:
            final_segments.append((x, y, w, h, region_color))

    # Sort final segments left-to-right.
    final_segments = sorted(final_segments, key=lambda seg: seg[0])

    # Create a subfolder for this image's segments.
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_folder = os.path.join(output_folder, image_name)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    # Optionally, draw bounding boxes on a copy of the original image.
    output_image = image.copy()
    for i, (X, Y, W, H, seg_img) in enumerate(final_segments):
        cv2.rectangle(output_image, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
        seg_path = os.path.join(image_output_folder, f"char_{i}.png")
        cv2.imwrite(seg_path, seg_img)
        print("Saved:", seg_path)

    # Save the annotated image.
    annotated_path = os.path.join(image_output_folder, "segmentation.png")
    cv2.imwrite(annotated_path, output_image)
    print("Saved annotated image:", annotated_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python automation.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # List supported image files from the input folder.
    supported_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
             if os.path.isfile(os.path.join(input_folder, f)) and os.path.splitext(f)[1].lower() in supported_ext]

    if not files:
        print("No images found in the input folder.")
        sys.exit(1)

    # Randomly pick 20 images from the folder (or all if less than 20).
    sample_count = 20 if len(files) >= 20 else len(files)
    sample_files = random.sample(files, sample_count)
    print(f"Selected {sample_count} random images for processing.")

    # Process each selected image.
    for file in sample_files:
        print("Processing image:", file)
        process_image(file, output_folder)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import sys
import random

# # Global parameters.
# MIN_AREA = 100  # Minimum area (in pixels) to consider a region as a valid character.
# WIDE_THRESHOLD_MULTIPLIER = 120  # If region width > 1.2x the median width, try to split it by color.
# K_CLUSTERS = 2  # Number of clusters for k-means clustering in color splitting.
# COLOR_SPLIT_MIN_RATIO = 0.005  # Minimum fraction of pixels (0.5%) for a cluster to be accepted.
#
#
# def load_image(filename):
#     image = cv2.imread(filename)
#     if image is None:
#         print("Error: Could not load image:", filename)
#         sys.exit(1)
#     return image
#
#
# def convert_to_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#
# def equalize_and_blur(gray_image):
#     equalized = cv2.equalizeHist(gray_image)
#     blurred = cv2.medianBlur(equalized, 3)
#     return blurred
#
#
# def binarize_image(blurred_image):
#     binary = cv2.adaptiveThreshold(
#         blurred_image, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return closed
#
#
# def find_contours(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours
#
#
# def get_bounding_boxes(contours):
#     boxes = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if w * h >= MIN_AREA:
#             boxes.append((x, y, w, h))
#     return boxes
#
#
# def sort_boxes_left_to_right(boxes):
#     return sorted(boxes, key=lambda b: b[0])
#
#
# def median_width(boxes):
#     widths = [box[2] for box in boxes]
#     if not widths:
#         return 0
#     return np.median(widths)
#
#
# def color_split_region(region_color):
#     """Attempt to split a wide region using k-means clustering in HSV color space."""
#     hsv_region = cv2.cvtColor(region_color, cv2.COLOR_BGR2HSV)
#     pixel_values = hsv_region.reshape((-1, 3))
#     pixel_values = np.float32(pixel_values)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     ret, labels, centers = cv2.kmeans(pixel_values, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     labels = labels.flatten()
#     labels_image = labels.reshape((region_color.shape[0], region_color.shape[1]))
#     total_pixels = region_color.shape[0] * region_color.shape[1]
#     segments = []
#     for k in range(K_CLUSTERS):
#         mask = np.uint8((labels_image == k) * 255)
#         if cv2.countNonZero(mask) < COLOR_SPLIT_MIN_RATIO * total_pixels:
#             continue
#         kernel = np.ones((3, 3), np.uint8)
#         mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#         cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if cnts:
#             largest_contour = max(cnts, key=cv2.contourArea)
#             sx, sy, sw, sh = cv2.boundingRect(largest_contour)
#             if sw * sh >= MIN_AREA:
#                 segment = region_color[sy:sy + sh, sx:sx + sw]
#                 segments.append((sx, sy, sw, sh, segment))
#     return segments
#
#
# def process_image(image_path):
#     """
#     Process a single image: load, preprocess, segment characters,
#     and draw bounding boxes on the original image.
#     Returns the annotated image.
#     """
#     image = load_image(image_path)
#     gray = convert_to_grayscale(image)
#     blurred = equalize_and_blur(gray)
#     binary = binarize_image(blurred)
#     contours = find_contours(binary)
#     boxes = get_bounding_boxes(contours)
#     sorted_boxes = sort_boxes_left_to_right(boxes)
#     med_width = median_width(sorted_boxes)
#
#     final_segments = []
#     for box in sorted_boxes:
#         x, y, w, h = box
#         region_color = image[y:y + h, x:x + w]
#         # For regions wider than expected, try splitting by color.
#         if w > (WIDE_THRESHOLD_MULTIPLIER / 100.0) * med_width:
#             splits = color_split_region(region_color)
#             if len(splits) > 1:
#                 for (sx, sy, sw, sh, seg_img) in splits:
#                     final_segments.append((x + sx, y + sy, sw, sh))
#             else:
#                 final_segments.append((x, y, w, h))
#         else:
#             final_segments.append((x, y, w, h))
#
#     # Draw bounding boxes on a copy of the original image.
#     annotated = image.copy()
#     for (X, Y, W, H) in final_segments:
#         cv2.rectangle(annotated, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
#     return annotated
#
#
# def main():
#
#     input_folder = sys.argv[1]
#     output_folder = sys.argv[2]
#
#     # List only image files from the input folder.
#     supported_ext = [".png"]
#     files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
#              if os.path.isfile(os.path.join(input_folder, f)) and os.path.splitext(f)[1].lower() in supported_ext]
#
#     if not files:
#         print("No images found in the input folder.")
#         sys.exit(1)
#
#     # Randomly pick 20 images (or fewer if not available).
#     sample_count = 20 if len(files) >= 20 else len(files)
#     sample_files = random.sample(files, sample_count)
#     print(f"Selected {sample_count} random images for processing.")
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for file in sample_files:
#         print("Processing image:", file)
#         annotated = process_image(file)
#         base_name = os.path.splitext(os.path.basename(file))[0]
#         out_path = os.path.join(output_folder, base_name + "_segmentation.png")
#         cv2.imwrite(out_path, annotated)
#         print("Saved segmented output image:", out_path)
#
#
# if __name__ == "__main__":
#     main()
# !/usr/bin/env python3
import cv2
import numpy as np
import os
import sys

# # Parameters for contour area filtering and normalization
# MIN_AREA = 10  # Minimum area (pixels) for a valid character
# MAX_AREA = 5000  # Maximum area (pixels) for a valid character
# KERNEL_SIZE = (3, 3)  # Kernel size for morphological closing
# OUTPUT_SIZE = (28, 28)  # Standard size to which each character is resized
#
#
# def load_image(filepath):
#     image = cv2.imread(filepath)
#     if image is None:
#         print(f"Error: Could not load image {filepath}")
#         sys.exit(1)
#     return image
#
#
# def preprocess_image(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Equalize histogram for contrast enhancement
#     equalized = cv2.equalizeHist(gray)
#     # Apply median blur to reduce noise
#     blurred = cv2.medianBlur(equalized, 3)
#     # Binarize the image using adaptive thresholding (white text on black background)
#     binary = cv2.adaptiveThreshold(
#         blurred, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2)
#     # Morphological closing (3x3 kernel) to fill small gaps in characters
#     kernel = np.ones(KERNEL_SIZE, np.uint8)
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
#     return closed
#
#
# def find_valid_contours(binary):
#     # Find contours in the binarized image
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     valid_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if MIN_AREA <= area <= MAX_AREA:
#             valid_contours.append(cnt)
#     return valid_contours
#
#
# def sort_contours_left_to_right(contours):
#     # Sort contours based on the x-coordinate of their bounding boxes
#     bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
#     sorted_indices = np.argsort([box[0] for box in bounding_boxes])
#     sorted_contours = [contours[i] for i in sorted_indices]
#     return sorted_contours
#
#
# def extract_and_normalize_characters(binary, contours):
#     tokens = []
#     bounding_boxes = []
#     # For each contour, crop and resize the character region
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         bounding_boxes.append((x, y, w, h))
#         # Crop from the binary image
#         char_img = binary[y:y + h, x:x + w]
#         # Resize to the standard size (28x28)
#         norm_img = cv2.resize(char_img, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
#         # Optionally: normalize pixel values to [0,1] if used later in training
#         # norm_img = norm_img.astype("float32") / 255.0
#         tokens.append(norm_img)
#     return tokens, bounding_boxes
#
#
# def annotate_image(image, bounding_boxes):
#     annotated = image.copy()
#     for (x, y, w, h) in bounding_boxes:
#         cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return annotated
#
#
# def process_single_image(filepath):
#     image = load_image(filepath)
#     # Preprocess to get a binary image with closed character regions
#     binary = preprocess_image(image)
#     # Find contours and filter based on area
#     contours = find_valid_contours(binary)
#     # Sort contours from left to right
#     sorted_contours = sort_contours_left_to_right(contours)
#     # Extract and normalize each character
#     tokens, bounding_boxes = extract_and_normalize_characters(binary, sorted_contours)
#     # Create an annotated image with bounding boxes drawn on the original image
#     annotated = annotate_image(image, bounding_boxes)
#     return tokens, bounding_boxes, annotated
#
#
# def main():
#     if len(sys.argv) < 3:
#         print("Usage: python segmentation_tokenization.py <input_folder> <output_folder>")
#         sys.exit(1)
#
#     input_folder = sys.argv[1]
#     output_folder = sys.argv[2]
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # List image files in the input folder (supports common image formats)
#     supported_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
#     files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
#              if os.path.isfile(os.path.join(input_folder, f)) and os.path.splitext(f)[1].lower() in supported_ext]
#
#     for file in files:
#         print(f"Processing {file} ...")
#         tokens, bounding_boxes, annotated = process_single_image(file)
#         base_name = os.path.splitext(os.path.basename(file))[0]
#         # Save the annotated image (with bounding boxes)
#         annotated_path = os.path.join(output_folder, f"{base_name}_annotated.png")
#         cv2.imwrite(annotated_path, annotated)
#         # Save each tokenized (normalized) character image
#         for idx, token in enumerate(tokens):
#             token_path = os.path.join(output_folder, f"{base_name}_char_{idx}.png")
#             cv2.imwrite(token_path, token)
#         print(f"Saved annotated image and {len(tokens)} token images for {base_name}")
#
#
# if __name__ == "__main__":
#     main()

