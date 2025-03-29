import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Parameters
img_w, img_h = 128, 64   # width=128, height=64
batch_size = 32
max_label_len = 10
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
num_classes = len(characters) + 1  # one extra for the CTC blank token

# Mappings
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}

def encode_label(text):
    return [char_to_num[c] for c in text]

def data_generator(data_dir, batch_size):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
    np.random.shuffle(files)
    while True:
        X_data, labels, input_lengths, label_lengths = [], [], [], []

        for f in files:
            img_path = os.path.join(data_dir, f)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, [img_h, img_w])
            img = tf.cast(img, tf.float32) / 255.0
            X_data.append(img.numpy())

            label_str = os.path.splitext(f)[0]
            encoded = encode_label(label_str)
            if len(encoded) > max_label_len:
                print(f"Warning: Label '{label_str}' too long, truncating.")
                encoded = encoded[:max_label_len]

            labels.append(encoded)
            label_lengths.append(len(encoded))
            input_lengths.append(32)  # CNN output time steps (adjust if necessary)

            if len(X_data) == batch_size:
                X_data = np.array(X_data)
                # Use a padding value that is not a valid label (-1 instead of 0)
                labels_padded = keras.preprocessing.sequence.pad_sequences(
                    labels, maxlen=max_label_len, padding='post', value=-1
                )
                yield (
                    {
                        'input_image': X_data,
                        'labels': np.array(labels_padded),
                        'input_length': np.array(input_lengths).reshape(-1, 1),
                        'label_length': np.array(label_lengths).reshape(-1, 1)
                    },
                    np.zeros([batch_size])
                )
                X_data, labels, input_lengths, label_lengths = [], [], [], []

# --- CRNN Model ---
input_image = keras.Input(shape=(img_h, img_w, 1), name='input_image')
labels_input = keras.Input(name='labels', shape=(max_label_len,), dtype='int32')
input_length = keras.Input(name='input_length', shape=(1,), dtype='int32')
label_length = keras.Input(name='label_length', shape=(1,), dtype='int32')

# CNN backbone
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_image)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 1))(x)

x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 1))(x)

# Prepare for RNN: (batch, width, height, channels) -> (batch, time_steps, features)
x = layers.Permute((2, 1, 3))(x)  # Now shape: (batch, time_steps, height, channels)
x = layers.Reshape((32, 4 * 512))(x)  # 32 is the time_steps (adjust if necessary)

x = layers.Dense(64, activation='relu')(x)

# RNN layers
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# Output layer
y_pred = layers.Dense(num_classes, activation='softmax')(x)

# CTC Loss Function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [y_pred, labels_input, input_length, label_length]
)

# Build and compile model
model = keras.models.Model(
    inputs=[input_image, labels_input, input_length, label_length],
    outputs=ctc_loss
)

model.summary()

# --- Accuracy Tracking Callback ---
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use CTC decode with greedy search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results.numpy():
        # Ignore the padding value (-1) during decoding
        text = ''.join([num_to_char.get(i, '') for i in res if i != -1])
        output_text.append(text)
    return output_text

class CTCCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_dir, batch_size, num_batches=5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.pred_model = None

    def on_train_begin(self, logs=None):
        # Build a model that outputs y_pred only (for prediction)
        self.pred_model = keras.Model(inputs=input_image, outputs=y_pred)

    def on_epoch_end(self, epoch, logs=None):
        gen = data_generator(self.data_dir, self.batch_size)
        total_strings, correct_strings = 0, 0
        total_chars, correct_chars = 0, 0

        for _ in range(self.num_batches):
            batch = next(gen)
            batch_images = batch[0]['input_image']
            true_labels = batch[0]['labels']

            preds = self.pred_model.predict(batch_images)
            decoded_preds = decode_batch_predictions(preds)

            for i, pred_text in enumerate(decoded_preds):
                # Reconstruct true text, ignoring padding (-1)
                true_text = ''.join([num_to_char.get(ch, '') for ch in true_labels[i] if ch != -1])
                if pred_text == true_text:
                    correct_strings += 1
                total_strings += 1

                # Calculate character-level accuracy
                match_len = min(len(pred_text), len(true_text))
                correct_chars += sum(1 for a, b in zip(pred_text, true_text) if a == b)
                total_chars += len(true_text)

        string_acc = correct_strings / total_strings
        char_acc = correct_chars / total_chars if total_chars > 0 else 0

        print(f"\n✅ String Accuracy: {string_acc:.2%} | 🔠 Character Accuracy: {char_acc:.2%}")

# --- Training ---
data_dir = '../data/preprocessed_images'
train_gen = data_generator(data_dir, batch_size)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss={'ctc': lambda y_true, y_pred: y_pred}
)

model.fit(
    train_gen,
    steps_per_epoch=200,
    epochs=50,
    callbacks=[
        CTCCallback(data_dir, batch_size),
        keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
)
