import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from data_processing import DigitProcessor
from digits_3d.utils import plot_confusion_matrix
from utils import get_distance

processor = DigitProcessor()

# Extracts the training data for the model
def get_training_data(training_data):
    all_strokes = processor.process_strokes(training_data)
    compact_data = processor.reduce_dims(all_strokes)
    features = compact_data[['pc1', 'pc2']].values
    labels = compact_data['label'].values
    return features, labels

def knn_classifier(x_test, k, x_train, y_train):
    n_train = x_train.shape[0]
    if len(x_test.shape) == 1:
        x_test = x_test.reshape(1, -1)  # Reshapes to 2D if it's 1D
    n_test = x_test.shape[0]

    label = np.zeros((n_test))
    d = np.zeros((n_test, n_train))

    for i in range(n_test):
        for j in range(n_train):
            min_features = min(x_test[i].shape[0], x_train[j].shape[0])
            d[i, j] = get_distance(x_test[i][:min_features], x_train[j][:min_features], type='euclidean')

        idx_knn = np.argsort(d[i])[:k]
        label[i] = np.bincount(y_train[idx_knn].astype(int)).argmax()

    return label.astype(int)

# Train the digit classifier
def train_digit_classifier(training_data):
    X, y = get_training_data(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    k = 7

    predictions = knn_classifier(X_test, k, X_train, y_train)
    with open('digit_classifier.pkl', 'wb') as f:  # saves the trained model
        pickle.dump((k, X_train, y_train), f)

    # Check model accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Classifier Accuracy: {acc:.2%}")
    conf_mat = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_mat)
    # Plot confusion matrix
    plot_confusion_matrix(conf_mat)


def get_classifier():
    if not os.path.exists('digit_classifier.pkl'):
        print("Model doesn't exist. Training new digit classifier...")
        train_digit_classifier()

    with open('digit_classifier.pkl', 'rb') as f:
        k, X_train, y_train = pickle.load(f)

    return k, X_train, y_train

# Predict a given stroke
def predict_digit(stroke_file):
    k, X_train, y_train = get_classifier()
    test_features = processor.process_and_extract_features(stroke_file)

    # predict the digit
    predictions = knn_classifier(test_features, k, X_train, y_train)
    digit = predictions[0]
    print(f"The given stroke is most likely: {digit}")
