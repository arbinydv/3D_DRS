import warnings

import numpy as np
from sklearn.metrics import confusion_matrix

from digits_3d.cnn_model import build_cnn_model
from digits_3d.data_processor import DataProcessor
import matplotlib.pyplot as plt

from digits_3d.utils import plot_confusion_matrix, plot_training_history


def train_model():
    warnings.filterwarnings("ignore")
    data_loader = DataProcessor(data_dir="training_data", n_resample=16)
    X, y = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split(X, y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Build the CNN model
    cnn_model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))

    # Train the model
    classifier = cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Test the model
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)

    y_pred_probs = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generates confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
    plot_training_history(classifier.history['accuracy'], classifier.history['val_accuracy'])

    print(f"Classifier Test Accuracy: {test_acc:.2%}")

    cnn_model.save("classifier_model.keras")


