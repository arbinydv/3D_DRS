import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def process_strokes():
    data = []
    scaler = MinMaxScaler()

    for digit in range(10):
        for file in Path('training_data').glob(f'stroke_{digit}_*.csv'):
            sample = int(file.stem.split('_')[-1])
            stroke = pd.read_csv(file, header=None, names=['x', 'y', 'z'])

            # Resample and normalize
            resampled = pd.DataFrame(signal.resample(stroke[['x', 'y']], 11), columns=['x', 'y'])
            normalized = pd.DataFrame(scaler.fit_transform(resampled), columns=['x', 'y'])

            # Compute unit vectors
            diff = normalized.diff().dropna()
            magnitude = np.linalg.norm(diff, axis=1)
            unit_vectors = diff.div(magnitude, axis=0)

            # Flatten and add metadata
            flattened = np.concatenate([unit_vectors.values.flatten(), [digit, sample]])
            data.append(flattened)

    columns = [f'f{i}' for i in range(len(data[0]) - 2)] + ['label', 'sample']
    return pd.DataFrame(data, columns=columns)


def reduce_dims(df, n_components=2):
    features = df.drop(columns=['label', 'sample'])
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)

    for i in range(n_components):
        df[f'pc{i + 1}'] = pca_features[:, i]

    return df
