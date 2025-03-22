import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Parameters
img_w, img_h = 128, 64
batch_size = 32
max_label_len = 10
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
num_classes = len(characters) + 1  # CTC blank

# Mappings
char_to_num = {char: i+1 for i, char in enumerate(characters)}
num_to_char = {i+1: char for i, char in enumerate(characters)}

def encode_label(text):
    return [char_to_num[c] for c in text]

def data_generator(data_dir, batch_size):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
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
            labels.append(encoded)
            label_lengths.append(len(encoded))
            input_lengths.append(img_w // 4)

            if len(X_data) == batch_size:
                X_data = np.array(X_data)
                labels_padded = keras.preprocessing.sequence.pad_sequences(
                    labels, maxlen=max_label_len, padding='post', value=0
                )
                yield ({
                    'input_image': X_data,
                    'labels': np.array(labels_padded),
                    'input_length': np.array(input_lengths).reshape(-1, 1),
                    'label_length': np.array(label_lengths).reshape(-1, 1)
                }, np.zeros([batch_size]))
                X_data, labels, input_lengths, label_lengths = [], [], [], []

# CRNN Model
input_image = keras.Input(shape=(img_h, img_w, 1), name='input_image')
labels_input = keras.Input(name='labels', shape=(max_label_len,), dtype='int32')
input_length = keras.Input(name='input_length', shape=(1,), dtype='int32')
label_length = keras.Input(name='label_length', shape=(1,), dtype='int32')

# CNN
x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_image)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(512, (2,2), padding='valid', activation='relu')(x)

# Reshape
conv_shape = x.shape
x = layers.Reshape((int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
x = layers.Dense(64, activation='relu')(x)

# RNN
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# Output
y_pred = layers.Dense(num_classes, activation='softmax')(x)

# CTC Loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [y_pred, labels_input, input_length, label_length]
)

# Model
model = keras.models.Model(
    inputs=[input_image, labels_input, input_length, label_length],
    outputs=ctc_loss
)

model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
model.summary()

# TRAINING
data_dir = '../data/preprocessed_images'
train_gen = data_generator(data_dir, batch_size)

model.fit(train_gen, steps_per_epoch=100, epochs=10)
