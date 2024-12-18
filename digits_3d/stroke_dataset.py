from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.model_selection import train_test_split



class StrokeDataset:
    def __init__(self, data_dir="training_data", n_resample=16):
        self.data_dir = Path(data_dir)
        self.n_resample = n_resample

    def _resample_and_normalize(self, file_path):
        # Read stroke data (x, y, z)
        stroke = pd.read_csv(file_path, header=None, names=['x', 'y', 'z'])
        stroke_resampled = signal.resample(stroke, self.n_resample)
        stroke_resampled = (stroke_resampled - stroke_resampled.min()) / (
                    stroke_resampled.max() - stroke_resampled.min())
        return stroke_resampled  # Resampled and normalized data

    def load_data(self):
        all_strokes = []
        labels = []

        for digit in range(10):  # Loop through digits 0-9
            for file in self.data_dir.glob(f"stroke_{digit}_*.csv"):
                features = self._resample_and_normalize(file)
                all_strokes.append(features)
                labels.append(digit)

        # Convert to numpy arrays
        X = np.array(all_strokes)
        y = np.array(labels)

        # Add a channel dimension for CNN input: (samples, points, 3, 1)
        X = X[..., np.newaxis]
        return X, y

    def split_data(self, X, y):
        # Split into 80% training and 20% testing
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)