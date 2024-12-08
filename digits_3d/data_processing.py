import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.decomposition import PCA
from digits_3d.utils import min_max_scaler, calculate_unit_vectors


class DigitProcessor:
    def __init__(self, n_resample=11, n_pca_components=5):
        self.n_resample = n_resample
        self.n_pca_components = n_pca_components
        self.pca = PCA(n_components=n_pca_components)
        self.feature_names = None

    def process_single_stroke(self, file_path, digit=None, sample=None):
        stroke = pd.read_csv(file_path, header=None, names=['x', 'y', 'z'])
        resampled = self._resample(stroke[['x', 'y']])
        normalized = self._normalize(resampled)
        unit_vectors = calculate_unit_vectors(normalized)
        features = unit_vectors.values.flatten()

        if digit is not None and sample is not None:
            features = np.concatenate([features, [digit, sample]])

        return features

    def process_strokes(self, data_dir='training_data'):
        processed_strokes = []
        for digit in range(10):
            for file in Path(data_dir).glob(f'stroke_{digit}_*.csv'):
                sample = int(file.stem.split('_')[-1])
                features = self.process_single_stroke(file, digit, sample)
                processed_strokes.append(features)

        columns = [f'f{i}' for i in range(len(processed_strokes[0]) - 2)] + ['label', 'sample']
        return pd.DataFrame(processed_strokes, columns=columns)

    def reduce_dims(self, df):
        features = df.drop(columns=['label', 'sample'])
        self.feature_names = features.columns  # Store feature names
        pca_features = self.pca.fit_transform(features)

        for i in range(self.n_pca_components):
            df[f'pc{i + 1}'] = pca_features[:, i]

        return df
    def process_and_extract_features(self, stroke_file):
        features = self.process_single_stroke(stroke_file)

        if self.feature_names is None:
            raise ValueError("PCA has not been fitted. Call reduce_dims() first.")

        features_df = pd.DataFrame([features[:len(self.feature_names)]], columns=self.feature_names)

        return self.pca.transform(features_df)

    def _resample(self, data):
        return pd.DataFrame(signal.resample(data, self.n_resample), columns=data.columns)

    def _normalize(self, data):
        return pd.DataFrame(min_max_scaler(data), columns=data.columns)
