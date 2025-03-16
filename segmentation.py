import cv2
import numpy as np
import os
import sys


# Constants
MIN_AREA = 100  # Minimum area (in pixels) to consider a region as a valid character.
WIDE_THRESHOLD_MULTIPLIER = 120  # If region width > 1.2x median width, split it by color.
K_CLUSTERS = 2  # Number of clusters for k-means clustering in color splitting.
COLOR_SPLIT_MIN_RATIO = 0.005  # Minimum fraction of pixels (0.5%) for a cluster to be accepted.

original_image = None


def load_image(filename):
    """ Load an image and handle errors. """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    image = cv2.imread(filename)
    if image is None:
        print(f"Error: Unable to load image {filename}. Check file format and integrity.")
        sys.exit(1)
    
    return image


def convert_to_grayscale(image):
    """ Convert image to grayscale """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def equalize_and_blur(gray_image):
    """ Equalize histogram and apply median blur to remove noise """
    equalized = cv2.equalizeHist(gray_image)
    blurred = cv2.medianBlur(equalized, 3)
    return blurred


def binarize_image(blurred_image):
    """ Apply adaptive thresholding to obtain a binary image """
    binary = cv2.adaptiveThreshold(
        blurred_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return closed


def apply_watershed(image):
    """ Applies Watershed Algorithm for better segmentation """

    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    image_copy = image.copy()
    cv2.watershed(image_copy, markers)

    image_copy[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return image_copy, markers


def extract_characters(image, markers):
    """ Extracts characters based on watershed segmentation. """
    extracted_chars = []

    for marker_id in np.unique(markers):
        if marker_id <= 1:  # Ignore background
            continue

        mask = np.uint8(markers == marker_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= MIN_AREA:  # Filter small areas
                char_segment = image[y:y+h, x:x+w]
                extracted_chars.append((x, y, w, h, char_segment))

    return extracted_chars


def process_and_display():
    """ Main processing function. """
    global original_image

    gray = convert_to_grayscale(original_image)
    blurred = equalize_and_blur(gray)
    binary = binarize_image(blurred)

    watershed_result, markers = apply_watershed(original_image)
    
    extracted_chars = extract_characters(original_image, markers)

    cv2.imshow("Watershed Segmentation", watershed_result)

    output_folder = "output_segments"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_image = original_image.copy()
    for i, (x, y, w, h, seg_img) in enumerate(extracted_chars):
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out_path = os.path.join(output_folder, f"char_{i}.png")
        cv2.imwrite(out_path, seg_img)
        print("Saved:", out_path)

    cv2.imshow("Segmented Characters", output_image)
    cv2.imwrite("output_watershed.png", watershed_result)  # Save final result

    print("Watershed segmentation applied. Press ESC to exit.")
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def main():
    """ Main entry point of the script. """
    global original_image

    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <image_file>")
        sys.exit(1)

    # Load the image
    original_image = load_image(sys.argv[1])

    # Process and display results
    process_and_display()


if __name__ == "__main__":
    main()
