import tensorflow as tf
import numpy as np
import os
import warnings
from data_preprocessor import load_data,split, data_preprocessing


def predict_digit(sample_data):
    warnings.filterwarnings("ignore")
    # Normalize the sample_data to fit with the model standard

    cleaned_sample = data_preprocessing(sample_data)
    test_sample = cleaned_sample[np.newaxis,...,np.newaxis]

    model_path="cnn_model.keras"

    # fetch the trained model if not the run the train_cnn model to train the model
    if not os.path.exists(model_path):
        print('Model not found. Building and Training the model....')
        train_model()
    cnn_classifier = tf.keras.models.load_model(model_path)

    prediction = np.argmax(cnn_classifier.predict(test_sample),axis=1)

    return prediction


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

    cnn_model.save("cnn_model.keras")


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

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
