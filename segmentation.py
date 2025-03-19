import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# Default parameters
params = {
    "use_gabor": False,   # Apply Gabor filtering?
    "invert": False,      # Invert colors (black text on white)
    "min_area": 50,       # Minimum area for character detection
    "max_area": 1000,     # Maximum area before letters are merged
    "kernel_size": 2,     # Morphological kernel size
}

# File handling
image_list = []
current_index = 0
input_folder = "preprocessed_images"
output_folder = "segmented"

def apply_gabor_filter(image):
    """Applies a Gabor filter to enhance edges."""
    kernel = cv2.getGaborKernel((20, 20), 3.0, 0, 1.0, 0.9, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)

def segment_characters(image):
    """Segment characters using current parameters."""
    global params

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Apply Gabor if enabled
    if params["use_gabor"]:
        gray = apply_gabor_filter(gray)

    # Thresholding
    thresh_type = cv2.THRESH_BINARY_INV if params["invert"] else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Morphological processing
    kernel = np.ones((params["kernel_size"], params["kernel_size"]), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter bounding boxes
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < params["min_area"]:
            continue
        if params["max_area"] > 0 and area > params["max_area"]:
            continue
        bounding_boxes.append((x, y, w, h))

    bounding_boxes.sort(key=lambda box: box[0])

    # Extract characters
    character_images = []
    segmented_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(segmented_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        char_img = processed[y:y+h, x:x+w]
        character_images.append(char_img)

    return segmented_img, character_images

def update_segmentation():
    """Update the segmentation preview with new parameters."""
    global image_list, current_index, input_folder

    if not image_list:
        return

    img_path = os.path.join(input_folder, image_list[current_index])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return

    segmented_img, _ = segment_characters(image)
    
    # Show original and segmented images side by side
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Image", segmented_img)

def load_images():
    """Load images from folder."""
    global image_list, current_index, input_folder
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    if input_folder:
        image_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_index = 0
        update_segmentation()

def next_image():
    """Move to the next image."""
    global current_index
    if image_list:
        current_index = (current_index + 1) % len(image_list)
        update_segmentation()

def prev_image():
    """Move to the previous image."""
    global current_index
    if image_list:
        current_index = (current_index - 1) % len(image_list)
        update_segmentation()

def save_segmented():
    """Save segmented characters."""
    global image_list, current_index, output_folder
    if not image_list:
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_path = os.path.join(input_folder, image_list[current_index])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return

    _, char_images = segment_characters(image)

    for idx, char_img in enumerate(char_images):
        out_filename = f"{os.path.splitext(image_list[current_index])[0]}_char_{idx}.png"
        out_path = os.path.join(output_folder, out_filename)
        cv2.imwrite(out_path, char_img)
        print(f"Saved segmented character: {out_path}")

def create_ui():
    """Creates the GUI control panel."""
    root = tk.Tk()
    root.title("CAPTCHA Segmentation Control Panel")

    # Load button
    load_btn = tk.Button(root, text="Load Images", command=load_images)
    load_btn.pack(pady=5)

    # Parameter sliders
    def update_param(key, value):
        params[key] = int(value) if key in ["min_area", "max_area", "kernel_size"] else bool(int(value))
        update_segmentation()

    sliders = {
        "Min Area": ("min_area", 10, 300, 50),
        "Max Area": ("max_area", 100, 3000, 1000),
        "Kernel Size": ("kernel_size", 1, 5, 2),
    }

    for label, (key, min_val, max_val, default) in sliders.items():
        frame = tk.Frame(root)
        frame.pack(pady=5)
        tk.Label(frame, text=label).pack(side=tk.LEFT)
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, command=lambda v, k=key: update_param(k, v))
        slider.set(default)
        slider.pack(side=tk.RIGHT)

    # Checkboxes
    checkboxes = {
        "Use Gabor Filter": "use_gabor",
        "Invert Colors": "invert"
    }

    for label, key in checkboxes.items():
        var = tk.IntVar(value=params[key])
        chk = tk.Checkbutton(root, text=label, variable=var, command=lambda k=key, v=var: update_param(k, v.get()))
        chk.pack(pady=5)

    # Navigation Buttons
    nav_frame = tk.Frame(root)
    nav_frame.pack(pady=5)
    prev_btn = tk.Button(nav_frame, text="Previous", command=prev_image)
    prev_btn.pack(side=tk.LEFT, padx=5)
    next_btn = tk.Button(nav_frame, text="Next", command=next_image)
    next_btn.pack(side=tk.RIGHT, padx=5)

    save_btn = tk.Button(root, text="Save Characters", command=save_segmented)
    save_btn.pack(pady=10)

    root.mainloop()

# Run the UI
create_ui()
