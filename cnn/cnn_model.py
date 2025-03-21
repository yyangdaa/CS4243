import os
import cv2
import numpy as np
from scipy import signal

###############################################################################
# 1) Layer base class
###############################################################################
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


###############################################################################
# 2) Activation base + ReLU / Softmax
###############################################################################
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


# ReLU activation
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        def relu_prime(x):
            return (x > 0).astype(x.dtype)  # derivative is 1 where x>0, else 0
        super().__init__(relu, relu_prime)


class Softmax(Layer):
    def forward(self, input):
        # Shift values for numerical stability
        shiftx = input - np.max(input)
        exps = np.exp(shiftx)
        self.output = exps / np.sum(exps)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Jacobian of Softmax
        # output_gradient shape: (num_classes, 1)
        n = np.size(self.output)
        # diag - outer product
        jacobian = np.diag(self.output.flatten()) - np.outer(self.output, self.output)
        return np.dot(jacobian, output_gradient)


###############################################################################
# 3) Dense layer
###############################################################################
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # weights: (output_size, input_size)
        self.weights = np.random.randn(output_size, input_size) * 0.01
        # bias: (output_size, 1)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # output_gradient shape: (output_size, 1)
        # input shape: (input_size, 1)
        # weights shape: (output_size, input_size)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Update
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


###############################################################################
# 4) Convolutional layer
###############################################################################
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        input_shape: (in_depth, in_height, in_width)
        kernel_size: int (square kernel)
        depth: number of output filters
        """
        super().__init__()
        in_depth, in_height, in_width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.kernel_size = kernel_size

        # Output shape for 'valid' correlation
        out_height = in_height - kernel_size + 1
        out_width = in_width - kernel_size + 1
        self.output_shape = (depth, out_height, out_width)

        # Kernels shape: (depth, in_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(depth, in_depth, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
        # output starts as a copy of biases
        self.output = np.copy(self.biases)

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
        # Prepare gradients
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)

        for d_out in range(self.depth):
            for d_in in range(self.input_shape[0]):
                # Gradient wrt. kernels
                kernels_gradient[d_out, d_in] = signal.correlate2d(
                    self.input[d_in],
                    output_gradient[d_out],
                    mode="valid"
                )
                # Gradient wrt. input
                input_gradient[d_in] += signal.convolve2d(
                    output_gradient[d_out],
                    self.kernels[d_out, d_in],
                    mode="full"
                )

        # Update
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


###############################################################################
# 5) MaxPooling2D layer
###############################################################################
class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size=2):
        """
        input_shape: (depth, height, width)
        pool_size: pool dimension (e.g., 2 for 2x2)
        """
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
        # input shape: (depth, height, width)
        self.input = input
        d, h, w = self.input_shape
        ps = self.pool_size

        out_d, out_h, out_w = self.output_shape
        output = np.zeros((out_d, out_h, out_w))
        self.argmax_indices = np.zeros((out_d, out_h, out_w, 2), dtype=np.int32)

        for depth_idx in range(d):
            for i in range(out_h):
                for j in range(out_w):
                    # define the 2D window
                    window = input[depth_idx,
                                   i*ps:(i+1)*ps,
                                   j*ps:(j+1)*ps]
                    # get max value and index
                    max_val = np.max(window)
                    output[depth_idx, i, j] = max_val
                    # store argmax location
                    local_max_idx = np.unravel_index(np.argmax(window), window.shape)
                    # global coords in input
                    self.argmax_indices[depth_idx, i, j] = (
                        i*ps + local_max_idx[0],
                        j*ps + local_max_idx[1]
                    )

        return output

    def backward(self, output_gradient, learning_rate):
        # output_gradient shape: (depth, out_height, out_width)
        d, h, w = self.output_shape
        ps = self.pool_size

        # gradient wrt input
        input_gradient = np.zeros_like(self.input)

        for depth_idx in range(d):
            for i in range(h):
                for j in range(w):
                    grad_val = output_gradient[depth_idx, i, j]
                    # find the location of the max
                    (orig_i, orig_j) = self.argmax_indices[depth_idx, i, j]
                    # route gradient back
                    input_gradient[depth_idx, orig_i, orig_j] += grad_val

        return input_gradient


###############################################################################
# 6) Reshape layer
###############################################################################
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


###############################################################################
# 7) Loss functions (categorical cross-entropy)
###############################################################################
def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred_clipped))

def categorical_cross_entropy_prime(y_true, y_pred):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return - (y_true / y_pred_clipped)


###############################################################################
# 8) Data loader for segmented CAPTCHA images
###############################################################################
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

def char_to_label(ch):
    return CHARSET.index(ch)

def load_segmented_images(folder_path):
    """
    Returns:
      X: numpy array (N, 1, 28, 28)
      Y: numpy array (N, 36, 1) one-hot
    """
    X_list = []
    Y_list = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue
        if "_char_" not in filename:
            continue

        # e.g. "0a1gfi_char_3"
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        if len(parts) != 3:
            continue

        captcha_str = parts[0]
        try:
            char_index = int(parts[2])
        except ValueError:
            continue
        if char_index < 0 or char_index >= len(captcha_str):
            continue

        label_char = captcha_str[char_index]
        try:
            label_idx = char_to_label(label_char)
        except ValueError:
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0
        # shape (1, 28, 28)
        img = np.reshape(img, (1, 28, 28))

        # one-hot label
        label_arr = np.zeros((len(CHARSET), 1), dtype=np.float32)
        label_arr[label_idx, 0] = 1.0

        X_list.append(img)
        Y_list.append(label_arr)

    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y


###############################################################################
# 9) Building an improved CNN
###############################################################################
def build_improved_cnn_model(num_classes=36):
    """
    Example CNN:
    - Conv( (1,28,28), kernel=3, depth=8 ) -> ReLU -> MaxPool(2x2) -> shape ~ (8, 13, 13)
    - Conv( (8,13,13), kernel=3, depth=16 ) -> ReLU -> MaxPool(2x2) -> shape ~ (16, 6, 6)
    - Reshape -> Dense(16*6*6, 128) -> ReLU -> Dense(128, num_classes) -> Softmax
    """
    model = []
    # 1) Conv -> ReLU -> MaxPool
    model.append(Convolutional((1, 28, 28), kernel_size=3, depth=8))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(8, 26, 26), pool_size=2))

    # 2) Conv -> ReLU -> MaxPool
    model.append(Convolutional((8, 13, 13), kernel_size=3, depth=16))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(16, 11, 11), pool_size=2))
    # shape after second MaxPool: (16, 5, 5) 
    # (Actually 11//2=5 if integer division, so final shape is (16,5,5).)

    # 3) Flatten
    model.append(Reshape((16, 5, 5), (16*5*5, 1)))  # 16*25=400

    # 4) Dense -> ReLU
    model.append(Dense(400, 128))
    model.append(ReLU())

    # 5) Dense -> Softmax
    model.append(Dense(128, num_classes))
    model.append(Softmax())
    return model


###############################################################################
# 10) Prediction and Mini-batch Training
###############################################################################
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(
    network,
    loss_func,
    loss_prime_func,
    x_train,
    y_train,
    epochs=10,
    learning_rate=0.01,
    batch_size=32,
    verbose=True
):
    """
    Mini-batch training:
     - Shuffle dataset
     - Split into batches
     - For each batch, do forward+backward on each sample (simple approach)
    """
    n_samples = len(x_train)
    for epoch in range(epochs):
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        total_error = 0.0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]

            # Train on each sample in this mini-batch
            for i in batch_inds:
                x = x_train[i]
                y = y_train[i]

                # forward
                output = predict(network, x)
                # loss
                total_error += loss_func(y, output)
                # backward
                grad = loss_prime_func(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

        avg_error = total_error / n_samples
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, loss={avg_error:.4f}")


###############################################################################
# 11) Main script: load data, build model, train, evaluate
###############################################################################
def main():
    segment_folder = "../output_segments/segmented"  # change as needed
    X, Y = load_segmented_images(segment_folder)

    if len(X) == 0:
        print("No valid segmented images found. Check folder or filenames.")
        return

    # Shuffle entire dataset once
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Train/test split
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test,  Y_test  = X[split:], Y[split:]

    print(f"Loaded {len(X)} samples.")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Build improved CNN
    model = build_improved_cnn_model(num_classes=len(CHARSET))

    # Train
    train(
        model,
        categorical_cross_entropy,
        categorical_cross_entropy_prime,
        X_train,
        Y_train,
        epochs=10,          # try even higher, like 20 or 30
        learning_rate=0.01, # tune the LR if needed
        batch_size=32,
        verbose=True
    )

    # Evaluate
    correct = 0
    for i in range(len(X_test)):
        y_pred = predict(model, X_test[i])
        pred_label = np.argmax(y_pred)
        true_label = np.argmax(Y_test[i])
        if pred_label == true_label:
            correct += 1
    accuracy = correct / len(X_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
