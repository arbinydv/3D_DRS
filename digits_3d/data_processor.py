import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from digits_3d.utils import get_distance, interpolate, min_max_normalization, pathify


class DataProcessor:
    def __init__(self, data_dir="training_data", n_resample=16):
        self.data_dir = Path(data_dir)
        self.n_resample = n_resample

    def resample_and_normalize(self, filename):
        stroke = pd.read_csv(filename, header=None, names=['x', 'y', 'z'])
        distances = [0]
        for i in range(1, len(stroke)):
            distances.append(distances[-1] + get_distance(stroke.iloc[i], stroke.iloc[i - 1], type='euclidean'))
        distances = np.array(distances)

        # creates evenly spaced distance
        even_distances = np.linspace(0, distances[-1], self.n_resample)

        resampled_stroke = np.column_stack([
            interpolate(distances, stroke[col], even_distances)
            for col in stroke.columns
        ])

        # min-max normalization
        normalized_stroke = min_max_normalization(resampled_stroke)

        return normalized_stroke

    def load_data(self):
        all_strokes = []
        labels = []
        # iterates through all training_data for 0-9 digits
        for digit in range(10):
            for file in self.data_dir.glob(f"stroke_{digit}_*.csv"):
                features = self.resample_and_normalize(file)
                all_strokes.append(features)
                labels.append(digit)

        X = np.array(all_strokes)
        y = np.array(labels)
        X = np.expand_dims(X, axis=-1)

        return X, y

    def split(self, X, y):
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)