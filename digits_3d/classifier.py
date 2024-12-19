import os.path
import warnings

import numpy as np
import tensorflow as tf
from digits_3d.data_processor import DataProcessor
from digits_3d.train_model import train_model

class DigitClassifier:
    def predict_digit(self, sample_data):
        warnings.filterwarnings("ignore")
        # Normalize the sample_data to fit with the model standard
        dataprocessor = DataProcessor()
        cleaned_sample = dataprocessor.resample_and_normalize(sample_data)
        test_sample = cleaned_sample[np.newaxis,...,np.newaxis]

        model_path="classifier_model.keras"

        # fetch the trained model if not the run the train_cnn model to train the model
        if not os.path.exists(model_path):
            print('Model not found. Training the model....')
            train_model()
        cnn_classifier = tf.keras.models.load_model(model_path)

        prediction = np.argmax(cnn_classifier.predict(test_sample),axis=1)

        return prediction
