import os
import cv2
import random
import numpy as np
from scipy import signal
from collections import defaultdict
from tqdm import tqdm  # progress bar import

# ===================== CONFIGURATION =====================
# Dataset
SEGMENT_FOLDER = "segmented"      # Folder containing segmented images
MAX_SAMPLES = 500                 # Increase if more labeled data is available

# Model Architecture
DENSE_UNITS = 128                 # Hidden layer size in the Dense layer

# Training
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
VERBOSE = True
# =========================================================

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        def relu_prime(x):
            return (x > 0).astype(x.dtype)
        super().__init__(relu, relu_prime)

class Softmax(Layer):
    def forward(self, input):
        shiftx = input - np.max(input)
        exps = np.exp(shiftx)
        self.output = exps / np.sum(exps)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Jacobian of the softmax
        jacobian = np.diag(self.output.flatten()) - np.outer(self.output, self.output)
        return np.dot(jacobian, output_gradient)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        in_depth, in_height, in_width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.kernel_size = kernel_size

        # Output shape for 'valid' correlation
        out_height = in_height - kernel_size + 1
        out_width = in_width - kernel_size + 1
        self.output_shape = (depth, out_height, out_width)

        # Initialize kernels and biases
        self.kernels = np.random.randn(depth, in_depth, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)  # start with bias

        in_depth, in_height, in_width = self.input_shape
        for d_out in range(self.depth):
            for d_in in range(in_depth):
                self.output[d_out] += signal.correlate2d(
                    self.input[d_in],
                    self.kernels[d_out, d_in],
                    mode="valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)

        for d_out in range(self.depth):
            for d_in in range(self.input_shape[0]):
                # Gradient wrt kernels
                kernels_gradient[d_out, d_in] = signal.correlate2d(
                    self.input[d_in],
                    output_gradient[d_out],
                    mode="valid"
                )
                # Gradient wrt input
                input_gradient[d_in] += signal.convolve2d(
                    output_gradient[d_out],
                    self.kernels[d_out, d_in],
                    mode="full"
                )

        # Update
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size=2):
        super().__init__()
        self.input_shape = input_shape
        self.pool_size = pool_size

        in_depth, in_height, in_width = input_shape
        out_height = in_height // pool_size
        out_width = in_width // pool_size
        self.output_shape = (in_depth, out_height, out_width)

        # Will store argmax indices for backward pass
        self.argmax_indices = None

    def forward(self, input):
        self.input = input
        d, h, w = self.input_shape
        ps = self.pool_size

        out_d, out_h, out_w = self.output_shape
        output = np.zeros((out_d, out_h, out_w))
        self.argmax_indices = np.zeros((out_d, out_h, out_w, 2), dtype=np.int32)

        for depth_idx in range(d):
            for i in range(out_h):
                for j in range(out_w):
                    window = input[depth_idx, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
                    max_val = np.max(window)
                    output[depth_idx, i, j] = max_val
                    local_max_idx = np.unravel_index(np.argmax(window), window.shape)
                    self.argmax_indices[depth_idx, i, j] = (
                        i*ps + local_max_idx[0],
                        j*ps + local_max_idx[1]
                    )
        return output

    def backward(self, output_gradient, learning_rate):
        d, h, w = self.output_shape
        input_gradient = np.zeros_like(self.input)

        for depth_idx in range(d):
            for i in range(h):
                for j in range(w):
                    grad_val = output_gradient[depth_idx, i, j]
                    (orig_i, orig_j) = self.argmax_indices[depth_idx, i, j]
                    input_gradient[depth_idx, orig_i, orig_j] += grad_val

        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))

def categorical_cross_entropy_prime(y_true, y_pred):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return - (y_true / y_pred_clipped)

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

def char_to_label(ch):
    return CHARSET.index(ch)

def load_segmented_images(folder_path, max_samples=500):
    """
    Loads up to 'max_samples' random images from the folder structure:
      folder_path/
        captcha_str_1/
          char_0.png
          char_1.png
          ...
        captcha_str_2/
          char_0.png
          ...
    """
    all_image_info = []
    for captcha_folder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, captcha_folder)
        if not os.path.isdir(subfolder_path):
            continue

        captcha_str = captcha_folder
        for filename in os.listdir(subfolder_path):
            if not filename.lower().endswith(".png"):
                continue
            if not filename.startswith("char_"):
                continue

            base = os.path.splitext(filename)[0]  # e.g. 'char_0'
            parts = base.split("_")               # ['char', '0']
            if len(parts) != 2:
                continue

            try:
                char_index = int(parts[1])
            except ValueError:
                continue

            if char_index < 0 or char_index >= len(captcha_str):
                continue

            label_char = captcha_str[char_index]
            if label_char not in CHARSET:
                # skip unknown chars
                continue

            img_path = os.path.join(subfolder_path, filename)
            all_image_info.append((captcha_str, char_index, img_path))

    if len(all_image_info) > max_samples:
        all_image_info = random.sample(all_image_info, max_samples)

    X_list = []
    Y_list = []
    meta_list = []

    for captcha_str, char_index, img_path in all_image_info:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0
        img = np.reshape(img, (1, 28, 28))

        label_idx = char_to_label(captcha_str[char_index])
        label_arr = np.zeros((len(CHARSET), 1), dtype=np.float32)
        label_arr[label_idx, 0] = 1.0

        X_list.append(img)
        Y_list.append(label_arr)
        meta_list.append({"captcha_str": captcha_str, "char_index": char_index})

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, meta_list

def build_improved_cnn_model(num_classes=36, dense_units=128):
    """
    Model:
      1) Conv((1,28,28), kernel=3, depth=8) -> ReLU -> MaxPool(2x2)
      2) Conv((8,13,13), kernel=3, depth=16) -> ReLU -> MaxPool(2x2)
      3) Reshape( (16,5,5) -> (400,1) )
      4) Dense(400, dense_units) -> ReLU
      5) Dense(dense_units, num_classes) -> Softmax
    """
    model = []
    model.append(Convolutional((1, 28, 28), kernel_size=3, depth=8))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(8, 26, 26), pool_size=2))

    model.append(Convolutional((8, 13, 13), kernel_size=3, depth=16))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(16, 11, 11), pool_size=2))

    model.append(Reshape((16, 5, 5), (16 * 5 * 5, 1)))  # Flatten => 400

    model.append(Dense(400, dense_units))
    model.append(ReLU())

    model.append(Dense(dense_units, num_classes))
    model.append(Softmax())
    return model

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# ------------------------- MODIFIED TRAIN FUNCTION -------------------------
def train(network, loss_func, loss_prime_func, x_train, y_train,
          x_val=None, y_val=None, meta_val=None, epochs=10, learning_rate=0.05, batch_size=64, verbose=True):
    """
    Simple mini-batch training:
      - Shuffle dataset
      - Split into batches
      - For each batch, perform forward+backward on each sample
    Additionally, if validation data is provided (x_val, y_val, meta_val), the function
    computes and prints character and string accuracy at the end of each epoch.
    """
    n_samples = len(x_train)
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        total_error = 0.0
        
        # Display progress bar for each batch
        for start_idx in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]

            for i in batch_inds:
                x = x_train[i]
                y = y_train[i]
                output = predict(network, x)
                total_error += loss_func(y, output)
                grad = loss_prime_func(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

        avg_error = total_error / n_samples
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, loss={avg_error:.4f}")

        # ----------- New code: compute validation accuracy -------------
        if x_val is not None and y_val is not None and meta_val is not None:
            # Compute Character Accuracy
            correct = 0
            for i in range(len(x_val)):
                y_pred = predict(network, x_val[i])
                pred_label = np.argmax(y_pred)
                true_label = np.argmax(y_val[i])
                if pred_label == true_label:
                    correct += 1
            char_accuracy = correct / len(x_val) * 100

            # Compute String Accuracy (Captcha prediction)
            captcha_predictions = defaultdict(dict)
            captcha_true = {}
            for i, meta_item in enumerate(meta_val):
                captcha_str = meta_item["captcha_str"]
                char_index = meta_item["char_index"]
                y_pred = predict(network, x_val[i])
                pred_char = CHARSET[np.argmax(y_pred)]
                captcha_predictions[captcha_str][char_index] = pred_char
                captcha_true[captcha_str] = captcha_str

            string_correct = 0
            for c_str, pred_dict in captcha_predictions.items():
                try:
                    predicted = ''.join(pred_dict[i] for i in range(len(c_str)))
                except KeyError:
                    predicted = ""
                if predicted == c_str:
                    string_correct += 1

            string_accuracy = string_correct / len(captcha_true) * 100

            if verbose:
                print(f"Validation - Char Accuracy: {char_accuracy:.2f}% | String Accuracy: {string_accuracy:.2f}%")
# ----------------------------------------------------------------------------

def main():
    # Load and shuffle dataset
    X, Y, meta = load_segmented_images(SEGMENT_FOLDER, max_samples=MAX_SAMPLES)
    if len(X) == 0:
        print("No valid segmented images found. Check folder or filenames.")
        return

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    meta = [meta[i] for i in indices]

    # Train/test split (80/20)
    split = int(0.8 * len(X))
    X_train, Y_train, meta_train = X[:split], Y[:split], meta[:split]
    X_test, Y_test, meta_test = X[split:], Y[split:], meta[split:]

    print(f"Loaded {len(X)} samples total.")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Build CNN with specified dense units
    model = build_improved_cnn_model(num_classes=len(CHARSET), dense_units=DENSE_UNITS)

    # Train the model using configuration parameters.
    # To show validation accuracy (character and string), we pass X_test, Y_test, and meta_test.
    train(
        model,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        X_train,
        Y_train,
        x_val=X_test,
        y_val=Y_test,
        meta_val=meta_test,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )

    # Final Evaluation (can be omitted if you prefer the per-epoch results)
    correct = 0
    for i in range(len(X_test)):
        y_pred = predict(model, X_test[i])
        pred_label = np.argmax(y_pred)
        true_label = np.argmax(Y_test[i])
        if pred_label == true_label:
            correct += 1
    char_accuracy = correct / len(X_test) * 100
    print(f"Final Character Accuracy: {char_accuracy:.2f}%")

    captcha_predictions = defaultdict(dict)
    captcha_true = {}
    for i, meta_item in enumerate(meta_test):
        captcha_str = meta_item["captcha_str"]
        char_index = meta_item["char_index"]
        y_pred = predict(model, X_test[i])
        pred_char = CHARSET[np.argmax(y_pred)]
        captcha_predictions[captcha_str][char_index] = pred_char
        captcha_true[captcha_str] = captcha_str

    string_correct = 0
    for c_str, pred_dict in captcha_predictions.items():
        try:
            predicted = ''.join(pred_dict[i] for i in range(len(c_str)))
        except KeyError:
            predicted = ""
        if predicted == c_str:
            string_correct += 1

    string_accuracy = string_correct / len(captcha_true) * 100
    print(f"Final String Accuracy: {string_accuracy:.2f}%")

if __name__ == "__main__":
    main()
