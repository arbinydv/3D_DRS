from digits_3d.stroke_dataset import StrokeDataset
import numpy as np
import tensorflow as tf


def digit_classify(model_path, file_path, n_resample=16):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Process the sample stroke
    processor = StrokeDataset(n_resample=n_resample)
    sample = processor._resample_and_normalize(file_path)
    sample = sample[np.newaxis, ..., np.newaxis]  # Reshape for CNN input

    # Predict the digit
    prediction = np.argmax(model.predict(sample), axis=1)
    print(f"Predicted digit: {prediction[0]}")


digit_classify("stroke_digit_cnn.keras", "training_data/stroke_6_0004.csv")