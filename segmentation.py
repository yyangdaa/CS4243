import cv2
import numpy as np
import os
import sys


MIN_AREA = 100  # Minimum area (in pixels) to consider a region as a valid character.
WIDE_THRESHOLD_MULTIPLIER = 120  # If region width > 1.2x the median width, we try to split it by color.
K_CLUSTERS = 2  # Number of clusters for k-means clustering in color splitting.
COLOR_SPLIT_MIN_RATIO = 0.005  # Minimum fraction of pixels (0.5%) for a cluster to be accepted.

original_image = None


def load_image(filename):
    image = cv2.imread(filename)
    if image is None:
        print("Error: Could not load image.")
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


def process_and_display():
    global original_image, MIN_AREA, WIDE_THRESHOLD_MULTIPLIER, K_CLUSTERS, COLOR_SPLIT_MIN_RATIO

    gray = convert_to_grayscale(original_image)
    cv2.imshow("Grayscale", gray)
    blurred = equalize_and_blur(gray)
    cv2.imshow("Blurred", blurred)
    binary = binarize_image(blurred)
    cv2.imshow("Binary", binary)

    contours = find_contours(binary)
    boxes = get_bounding_boxes(contours)
    sorted_boxes = sort_boxes_left_to_right(boxes)
    med_width = median_width(sorted_boxes)

    final_segments = []
    for box in sorted_boxes:
        x, y, w, h = box
        region_color = original_image[y:y + h, x:x + w]
        if w > (WIDE_THRESHOLD_MULTIPLIER / 100.0) * med_width:
            splits = color_split_region(region_color)
            if len(splits) > 1:
                for (sx, sy, sw, sh, seg_img) in splits:
                    final_segments.append((x + sx, y + sy, sw, sh, seg_img))
            else:
                final_segments.append((x, y, w, h, region_color))
        else:
            final_segments.append((x, y, w, h, region_color))

    final_segments = sorted(final_segments, key=lambda seg: seg[0])

    output_folder = "output_segments"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_image = original_image.copy()
    for i, (X, Y, W, H, seg_img) in enumerate(final_segments):
        cv2.rectangle(output_image, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
        out_path = os.path.join(output_folder, "char_{}.png".format(i))
        cv2.imwrite(out_path, seg_img)
        print("Saved:", out_path)

    cv2.imshow("Segmented Characters", output_image)


def update_parameters(val):
    global MIN_AREA, WIDE_THRESHOLD_MULTIPLIER, K_CLUSTERS, COLOR_SPLIT_MIN_RATIO

    MIN_AREA = cv2.getTrackbarPos("MIN_AREA", "Controls")
    WIDE_THRESHOLD_MULTIPLIER = cv2.getTrackbarPos("WIDE_MULTIPLIER", "Controls")
    K_CLUSTERS = cv2.getTrackbarPos("K_CLUSTERS", "Controls")
    ratio_val = cv2.getTrackbarPos("COLOR_MIN_RATIO", "Controls")
    COLOR_SPLIT_MIN_RATIO = ratio_val / 1000.0

    process_and_display()


def main():
    global original_image

    if len(sys.argv) < 2:
        print("Usage: python final.py <captcha_image>")
        sys.exit(1)

    original_image = load_image(sys.argv[1])

    if not os.path.exists("output_segments"):
        os.makedirs("output_segments")

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 200)

    cv2.createTrackbar("MIN_AREA", "Controls", 100, 1000, update_parameters)
    cv2.createTrackbar("WIDE_MULTIPLIER", "Controls", 120, 300, update_parameters)
    cv2.createTrackbar("K_CLUSTERS", "Controls", 2, 5, update_parameters)
    cv2.createTrackbar("COLOR_MIN_RATIO", "Controls", 5, 10, update_parameters)

    process_and_display()

    print("Press ESC to exit.")
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
