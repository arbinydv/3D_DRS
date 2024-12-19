import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Function to get distance (Manhattan or Euclidean)
def get_distance(x1, x2, type='euclidean'):
    if type == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    else:
        return np.sum(np.abs(x1 - x2))

# prepare input for CNN through interpolation to get new x, y, z values
def interpolate(x_values, y_values, x):
    return np.interp(x, x_values, y_values)

# applies min-max normalization to the input data, scales data between 0 and 1
def min_max_normalization(X, feature_range=(0, 1)):
    X = np.array(X)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_min, range_max = feature_range
    scaled_X = (X - min_vals) / (max_vals - min_vals)
    scaled_X = scaled_X * (range_max - range_min) + range_min
    return scaled_X

# Generate path
def generate_path(file_input):
    return os.path.join('training_data', file_input)

# Plots confusion matrix as a heatmap
def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix of Classifier Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_training_history(training, validation):
    plt.figure(figsize=(8, 4))
    plt.plot(training, label='Train Accuracy')
    plt.plot(validation, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()