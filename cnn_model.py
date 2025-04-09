import os
import cv2
import numpy as np
from scipy import signal
from tqdm import tqdm  # <-- Added for progress bars

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
        jacobian = np.diag(self.output.flatten()) - np.outer(self.output, self.output)
        return np.dot(jacobian, output_gradient)


###############################################################################
# 3) Dense layer
###############################################################################
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        # Initialize accumulators for mini-batch updates.
        self.reset_accumulators()

    def reset_accumulators(self):
        self.batch_weights_grad = np.zeros_like(self.weights)
        self.batch_bias_grad = np.zeros_like(self.bias)
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward_accumulate(self, output_gradient):
        # Compute gradients for the current sample.
        weights_grad = np.dot(output_gradient, self.input.T)
        bias_grad = output_gradient  # assuming bias gradient is just output_gradient

        # Accumulate gradients.
        self.batch_weights_grad += weights_grad
        self.batch_bias_grad += bias_grad

        # Return gradient w.r.t input to propagate backwards.
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient

    def update_parameters(self, learning_rate, batch_size):
        # Update parameters using the averaged gradients.
        self.weights -= learning_rate * (self.batch_weights_grad / batch_size)
        self.bias -= learning_rate * (self.batch_bias_grad / batch_size)
        self.reset_accumulators()

    # You can keep the old backward method for per-sample updates if needed:
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
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
        out_height = in_height - kernel_size + 1
        out_width = in_width - kernel_size + 1
        self.output_shape = (depth, out_height, out_width)
        self.kernels = np.random.randn(depth, in_depth, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(self.output_shape)

    def forward(self, input):
        self.input = input
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
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)
        for d_out in range(self.depth):
            for d_in in range(self.input_shape[0]):
                kernels_gradient[d_out, d_in] = signal.correlate2d(
                    self.input[d_in],
                    output_gradient[d_out],
                    mode="valid"
                )
                input_gradient[d_in] += signal.convolve2d(
                    output_gradient[d_out],
                    self.kernels[d_out, d_in],
                    mode="full"
                )
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
                    self.argmax_indices[depth_idx, i, j] = (i*ps + local_max_idx[0],
                                                            j*ps + local_max_idx[1])
        return output

    def backward(self, output_gradient, learning_rate):
        d, h, w = self.output_shape
        ps = self.pool_size
        input_gradient = np.zeros_like(self.input)
        for depth_idx in range(d):
            for i in range(h):
                for j in range(w):
                    grad_val = output_gradient[depth_idx, i, j]
                    (orig_i, orig_j) = self.argmax_indices[depth_idx, i, j]
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
    X_list = []
    Y_list = []
    captcha_info = []
    for captcha_name in os.listdir(folder_path):
        captcha_dir = os.path.join(folder_path, captcha_name)
        if not os.path.isdir(captcha_dir) or len(captcha_name) == 0:
            continue
        for filename in os.listdir(captcha_dir):
            if not filename.lower().endswith(".png") or not filename.startswith("char_"):
                continue
            try:
                char_index = int(filename.split("_")[1].split(".")[0])
            except:
                continue
            if char_index >= len(captcha_name):
                continue
            label_char = captcha_name[char_index]
            try:
                label_idx = char_to_label(label_char)
            except ValueError:
                continue
            img_path = os.path.join(captcha_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (28, 28))
            img = img.astype(np.float32) / 255.0
            img = np.reshape(img, (1, 28, 28))
            label_arr = np.zeros((len(CHARSET), 1), dtype=np.float32)
            label_arr[label_idx, 0] = 1.0
            X_list.append(img)
            Y_list.append(label_arr)
            captcha_info.append((captcha_name, char_index))
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y, captcha_info


###############################################################################
# 9) Building an improved CNN
###############################################################################
def build_improved_cnn_model(num_classes=36):
    model = []
    model.append(Convolutional((1, 28, 28), kernel_size=3, depth=8))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(8, 26, 26), pool_size=2))
    model.append(Convolutional((8, 13, 13), kernel_size=3, depth=16))
    model.append(ReLU())
    model.append(MaxPooling2D(input_shape=(16, 11, 11), pool_size=2))
    model.append(Reshape((16, 5, 5), (16*5*5, 1)))
    model.append(Dense(400, 128))
    model.append(ReLU())
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

def train_mini_batch(network, loss_func, loss_prime_func, x_train, y_train,
                     captcha_info_train, epochs=10, learning_rate=0.01, batch_size=32, verbose=True):
    n_samples = len(x_train)
    for epoch in range(epochs):
        # Shuffle indices at the start of each epoch.
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        total_loss = 0.0

        for start_idx in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]
            # Reset gradient accumulators for each parameterized layer.
            for layer in network:
                if hasattr(layer, 'reset_accumulators'):
                    layer.reset_accumulators()

            # Process each sample in the mini-batch.
            for i in batch_inds:
                x = x_train[i]
                y = y_train[i]

                # Forward pass.
                output = x
                for layer in network:
                    # For layers like Dropout that use a training flag.
                    if isinstance(layer, Dropout):
                        output = layer.forward(output, training=True)
                    else:
                        output = layer.forward(output)

                loss = loss_func(y, output)
                total_loss += loss

                # Compute initial gradient from loss and then backpropagate.
                grad = loss_prime_func(y, output)
                for layer in reversed(network):
                    # Instead of updating weights immediately, accumulate gradients.
                    if hasattr(layer, 'backward_accumulate'):
                        grad = layer.backward_accumulate(grad)
                    else:
                        # For layers without learnable parameters (or if you haven't implemented accumulation), use the old backward.
                        grad = layer.backward(grad, learning_rate)

            # End of mini-batch: update parameters for layers that support mini-batch updates.
            for layer in network:
                if hasattr(layer, 'update_parameters'):
                    layer.update_parameters(learning_rate, len(batch_inds))

        avg_loss = total_loss / n_samples
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            # Optionally, evaluate on training data.
            evaluate(network, x_train, y_train, captcha_info_train)



###############################################################################
# 11) Evaluation: Compute character and string (captcha) accuracy
###############################################################################
def evaluate(network, x_test, y_test, captcha_info):
    total_chars = len(x_test)
    correct_chars = 0
    captcha_predictions = {}
    for i in range(total_chars):
        output = predict(network, x_test[i])
        pred_idx = np.argmax(output)
        true_idx = np.argmax(y_test[i])
        if pred_idx == true_idx:
            correct_chars += 1
        captcha_name, char_index = captcha_info[i]
        predicted_char = CHARSET[pred_idx]
        true_char = captcha_name[char_index]
        if captcha_name not in captcha_predictions:
            captcha_predictions[captcha_name] = []
        captcha_predictions[captcha_name].append((char_index, predicted_char, true_char))
    char_accuracy = correct_chars / total_chars * 100
    correct_captchas = 0
    for captcha_name, predictions in captcha_predictions.items():
        predictions.sort(key=lambda x: x[0])
        predicted_string = "".join([p[1] for p in predictions])
        true_string = "".join([p[2] for p in predictions])
        if predicted_string == true_string:
            correct_captchas += 1
    string_accuracy = correct_captchas / len(captcha_predictions) * 100
    print(f"Character Accuracy: {char_accuracy:.2f}%")
    print(f"String Accuracy: {string_accuracy:.2f}%")

###############################################################################
# 12) Main script: load data, build model, train, and evaluate
###############################################################################
def main():
    segment_folder = "segmented"  # Change as needed.
    X, Y, captcha_info = load_segmented_images(segment_folder)
    if len(X) == 0:
        print("No valid segmented images found. Check folder or filenames.")
        return
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    captcha_info = [captcha_info[i] for i in indices]
    split = int(0.8 * n_samples)
    X_train, Y_train, captcha_info_train = X[:split], Y[:split], captcha_info[:split]
    X_test,  Y_test,  captcha_info_test  = X[split:], Y[split:], captcha_info[split:]
    print(f"Loaded {n_samples} samples.")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    model = build_improved_cnn_model(num_classes=len(CHARSET))
    # Pass captcha_info_train to the train function:
    train_mini_batch(model, categorical_cross_entropy, categorical_cross_entropy_prime, X_train, Y_train, captcha_info_train, epochs=10, learning_rate=0.01, batch_size=32, verbose=True)
    print("\nEvaluation on test set:")
    evaluate(model, X_test, Y_test, captcha_info_test)

if __name__ == "__main__":
    main()
