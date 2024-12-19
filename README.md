# Digit Recognition System
This project builds the model to understand and predict the 3D hand drawn digits using CNN classifier.

## Dataset and Algorithms
The dataset is obtained from the leapmotion company that works on building the VR and AR features.
We have used the CNN classifiers to train the model for the prediction. The model trains with the .csv strokes data
that represent coordinates in [0-9] range.

### Accuracy
Currently we have been able to achieve 94% of accuracy for the model and in the process of improving the model. 

### Libraries
```
• NumPy (np)
• Tenserflow (tf)
• Pandas (pd)  
• Pathlib  
• SciPy (scipy)  
• Scikit-learn  
• digits_3d.utils
```
### Algorithm 
```commandline 
# Entry Point
digit_classify()
```
### Future Prospects
We aim to use this model and integrate the front-end component that allows users to place the strokes
and see the prediction in the real time on the GUI.

