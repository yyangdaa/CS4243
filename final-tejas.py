#!/usr/bin/env python3
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
    """
    Split a wide region by performing k-means clustering in HSV color space.
    This helps in separating merged characters.
    """
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
        print("Usage: python segmentation.py <input_folder> <output_folder>")
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

    # Process each image in the folder.
    for file in files:
        print("Processing image:", file)
        process_image(file, output_folder)

if __name__ == "__main__":
    main()
