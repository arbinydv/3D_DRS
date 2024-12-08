# Digit Recognition System
This project builds the model to understand and predict the 3D hand drawn digits using KNN classifier.

## Dataset and Algorithms
The dataset is obtained from the leapmotion company that works on building the VR and AR features.
We have used the KNN classifiers to train the model for the prediction. The model trains with the .csv strokes data
that represent coordinates in [0-9] range.

### Accuracy
Currently we have been able to achieve 65% of accuracy for the model and in the process of improving the model. 

### Libraries
```
• NumPy (np)  
• Pandas (pd)  
• Pathlib  
• SciPy (scipy)  
• Scikit-learn  
• digits_3d.utils
```
### Algorithm 
```commandline
import numpy as np

def knn_predict(X_train, y_train, X_test, k=3):
    """
     KNN classifier wrapper to predict labels for test data.

    Parameters:
    - X_train: Training data (2D array)
    - y_train: Training labels (1D array)
    - X_test: Test data (2D array)
    - k: Number of nearest neighbors (default is 3)

    Returns:
    - Predictions: Predicted labels for test data
    """
    predictions = []
    for test_point in X_test:
        # Calculate distances from the test point to all train points
        distances = np.linalg.norm(X_train - test_point, axis=1)
        
        # Get indices of the k nearest neighbors
        k_neighbors = np.argsort(distances)[:k]
        
        # Get most common label among the neighbors
        neighbor_labels = y_train[k_neighbors]
        predicted_label = np.bincount(neighbor_labels).argmax()
        
        predictions.append(predicted_label)
    
    return np.array(predictions)
    
# function call 
predictions = knn_predict(X_train, y_train, X_test, k=3)
```
### Future Prospects
We aim to use this model to have 95+% accuracy an integrate the front-end component that allows users to place the strokes
and see the prediction in the real time on the GUI.

