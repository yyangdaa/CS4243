import cv2
import numpy as np
import os
import random


def apply_gabor_filter(image, ksize=20, sigma=3.0, theta=0, lambd=1.0, gamma=0.9, psi=0):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered


def segment_characters(image):
    # If image is colored, convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Optionally enhance edges with Gabor filter
    filtered = apply_gabor_filter(gray)

    # Threshold the image (assuming white text on black background)
    _, binary = cv2.threshold(filtered, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Dilate to get sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling for watershed (adding one to ensure background isn’t 0)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed requires a 3-channel image; convert if needed
    markers = cv2.watershed(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR), markers)

    # Collect segmented character regions (ignoring boundaries marked by -1)
    character_images = []
    for marker in range(2, ret + 2):  # markers start at 2 (1 is background)
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255

        # Find contours in the segmented mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Choose the largest contour in the region
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 100:
                char_img = binary[y:y + h, x:x + w]
                character_images.append(char_img)
    return character_images


def process_images(input_folder, output_folder, sample_size=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List image files (you can adjust supported extensions as needed)
    images_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images_list) == 0:
        print("No image files found in the input folder.")
        return

    sample_size = min(sample_size, len(images_list))
    sample_images = random.sample(images_list, sample_size)

    for img_file in sample_images:
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)



        char_images = segment_characters(image)
        if not char_images:
            continue

        for idx, char_img in enumerate(char_images):
            out_filename = os.path.splitext(img_file)[0] + f'_char_{idx}.png'
            out_path = os.path.join(output_folder, out_filename)
            cv2.imwrite(out_path, char_img)
            print(f"Saved segmented character: {out_path}")


if __name__ == "__main__":
    input_folder = "preprocessed_images"
    output_folder = "segmented"
    process_images(input_folder, output_folder, sample_size=20)
