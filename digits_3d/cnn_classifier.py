import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import get_distance, min_max_normalization, interpolate


def resample_and_normalize(filename):
    stroke = pd.read_csv(filename, header=None, names=['x', 'y', 'z'])
    distances = [0]
    for i in range(1, len(stroke)):
        distances.append(distances[-1] + get_distance(stroke.iloc[i], stroke.iloc[i - 1], type='euclidean'))
    distances = np.array(distances)

    # creates evenly spaced distance
    even_distances = np.linspace(0, distances[-1], 16)

    resampled_stroke = np.column_stack([
        interpolate(distances, stroke[col], even_distances)
        for col in stroke.columns
    ])

    # min-max normalization
    normalized_stroke = min_max_normalization(resampled_stroke)

    return normalized_stroke


def load_data():
    all_strokes = []
    labels = []
    data_dir= Path("training_data")

    # iterates through all training_data for 0-9 digits
    for digit in range(10):
        for file in data_dir.glob(f"stroke_{digit}_*.csv"):
            features = resample_and_normalize(file)
            all_strokes.append(features)
            labels.append(digit)

    X = np.array(all_strokes)
    y = np.array(labels)
    X = np.expand_dims(X, axis=-1)

    return X, y

def split( X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train_model():
    warnings.filterwarnings("ignore")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Build the CNN model
    cnn_model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))

    # Train the model
    cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Test the model
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)

    print(f"Classifier Test Accuracy: {test_acc:.2%}")

    cnn_model.save("classifier_model.keras")


def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 1), activation='relu'), # First convolution
        tf.keras.layers.MaxPooling2D((2, 1)),  # Pooling layer
        tf.keras.layers.Conv2D(32, (3, 1), activation='relu'),  # Second convolution
        tf.keras.layers.Flatten(),  # flatten
        tf.keras.layers.Dense(64, activation='relu'),  # connected layers during making dense
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# TODO : Build a custom optimizer
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
