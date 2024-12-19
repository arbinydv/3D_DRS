import os
import warnings
import numpy as np
import tensorflow as tf
from cnn_classifier import resample_and_normalize,train_model

# Entry point of the model
def digit_classify(sample_data):
    prediction = predict_digit(sample_data)
    return prediction[0]

def predict_digit(sample_data):
    warnings.filterwarnings("ignore")
    # Normalize the sample_data to fit with the model standard

    cleaned_sample = resample_and_normalize(sample_data)
    test_sample = cleaned_sample[np.newaxis,...,np.newaxis]

    model_path="classifier_model.keras"

    # fetch the trained model if not the run the train_cnn model to train the model
    if not os.path.exists(model_path):
        print('Model not found. Training the model....')
        train_model()
    cnn_classifier = tf.keras.models.load_model(model_path)

    prediction = np.argmax(cnn_classifier.predict(test_sample),axis=1)

    return prediction


if __name__ == '__main__':
    print("\nPredicting the digit............\n")
    print(digit_classify("training_data/stroke_2_0068.csv"))

