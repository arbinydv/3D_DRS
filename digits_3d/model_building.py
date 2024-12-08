import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

from data_processing import process_strokes, reduce_dims


def get_training_data():
    all_strokes = process_strokes()
    compact_data = reduce_dims(all_strokes)
    features = compact_data[['pc1', 'pc2']].values
    labels = compact_data['label'].values
    return features, labels


def train_digit_classifier():
    X, y = get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    with open('digit_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Check performance
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Classifier Accuracy: {acc:.2%}")
    conf_mat = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(conf_mat)


def get_classifier():
    if not os.path.exists('digit_classifier.pkl'):
        print("Model not found. Training new classifier...")
        train_digit_classifier()

    with open('digit_classifier.pkl', 'rb') as f:
        return pickle.load(f)


def predict_digit(stroke_file):
    clf = get_classifier()

    # Process stroke data
    stroke_data = pd.read_csv(stroke_file, header=None)
    stroke_data.columns = ['x', 'y', 'z']
    stroke_data['time'] = range(len(stroke_data))
    stroke_data = stroke_data[['x', 'y', 'time']]

    # Resample and normalize
    resampled = pd.DataFrame(signal.resample(stroke_data[['x', 'y']], 11), columns=['x', 'y'])
    normalized = (resampled - resampled.min()) / (resampled.max() - resampled.min())

    # Compute unit vectors
    diffs = normalized.diff().dropna()
    magnitudes = np.sqrt(diffs['x'] ** 2 + diffs['y'] ** 2)
    unit_vectors = diffs.div(magnitudes, axis=0)

    # Reduce dimensions
    if len(unit_vectors) > 1:
        pca = PCA(n_components=2)
        test_features = pca.fit_transform(unit_vectors)
    else:
        print("Warning: Not enough data points for PCA")
        test_features = unit_vectors.values

    # Make prediction
    digit = clf.predict(test_features)[0]
    print(f"The stroke likely represents the digit: {digit}")

