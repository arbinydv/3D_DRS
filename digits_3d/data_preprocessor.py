from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_distance, min_max_normalization, interpolate

def load_data():
    all_strokes = []
    labels = []
    data_dir= Path("training_data")

    # iterates through all training_data for 0-9 digits
    for digit in range(10):
        for file in data_dir.glob(f"stroke_{digit}_*.csv"):
            features = data_preprocessing(file)
            all_strokes.append(features)
            labels.append(digit)

    X = np.array(all_strokes)
    y = np.array(labels)
    X = np.expand_dims(X, axis=-1)

    return X, y

def split( X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def data_preprocessing(file_name):
    stroke = pd.read_csv(file_name, header=None, names=['x', 'y', 'z'])
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

