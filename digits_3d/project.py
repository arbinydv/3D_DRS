import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from scipy.spatial.distance import euclidean

# 1. Load the CSV data
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.columns = ['x', 'y', 'z']
    return data

# 2. Extract statistical features (enhanced)
def extract_features(data):
    """
    Extract features from 3D trajectory data:
    - Trajectory length
    - Curvature (angles between segments)
    - Velocity (mean and std)
    - Mean and std of x, y, z
    """
    coords = data.values
    features = []

    # Path length
    path_length = np.sum([euclidean(coords[i], coords[i+1]) for i in range(len(coords) - 1)])
    features.append(path_length)

    # Curvature (angles between segments)
    diff = np.diff(coords, axis=0)
    angles = []
    for i in range(len(diff) - 1):
        dot_product = np.dot(diff[i], diff[i+1])
        norm_product = np.linalg.norm(diff[i]) * np.linalg.norm(diff[i+1])
        if norm_product > 0:
            angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
            angles.append(angle)
    features.append(np.mean(angles) if angles else 0)
    features.append(np.std(angles) if angles else 0)

    # Velocity statistics
    velocities = [euclidean(coords[i], coords[i+1]) for i in range(len(coords) - 1)]
    features.append(np.mean(velocities))
    features.append(np.std(velocities))

    # Mean and std of x, y, z
    for axis in ['x', 'y', 'z']:
        features.append(np.mean(data[axis]))
        features.append(np.std(data[axis]))

    return features

# 3. Prepare training data
def prepare_training_data(csv_folder):
    features = []
    labels = []
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            label = int(filename.split('_')[1])
            data = load_data(os.path.join(csv_folder, filename))
            feature_vector = extract_features(data)
            features.append(feature_vector)
            labels.append(label)
    return np.array(features), np.array(labels)

# 4. Visualizations
def plot_accuracy_vs_time(accuracy_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, marker='o')
    plt.title("Model Accuracy Over Time")
    plt.xlabel("Training Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_3d_stroke(data, label):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['x'], data['y'], data['z'], label=f'Digit: {label}')
    ax.set_title(f'3D Trajectory for Digit {label}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# 5. Save and load model
def save_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# 6. Digit classification function
def digit_classify(testdata):
    model = load_model("trained_model.pkl")
    features = extract_features(testdata)
    return model.predict([features])[0]

# 7. Main function
def main():
    csv_folder = "training_data"  # Update to your folder path

    # Load and prepare the training data
    X, y = prepare_training_data(csv_folder)

    # Baseline accuracy
    baseline_accuracy = 1 / len(np.unique(y))
    print(f"Baseline Accuracy (Random Guess): {baseline_accuracy * 100:.2f}%")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Save the trained model
    save_model(knn, "trained_model.pkl")

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])

    # Plot accuracy (currently one point; track iterations if needed)
    plot_accuracy_vs_time([accuracy])

    # Visualize a random stroke's 3D trajectory
    random_idx = np.random.randint(len(y_test))
    random_file = os.listdir(csv_folder)[random_idx]
    random_data = load_data(os.path.join(csv_folder, random_file))
    plot_3d_stroke(random_data, y_test[random_idx])

if __name__ == "__main__":
    main()