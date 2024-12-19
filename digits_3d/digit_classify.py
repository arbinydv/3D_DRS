from digits_3d.classifier import DigitClassifier

# Entry point of the model
def digit_classify(sample_data):
    classifier = DigitClassifier()
    prediction = classifier.predict_digit(sample_data)
    return prediction[0]

if __name__ == '__main__':
    digit_classify("training_data/stroke_0_0002.csv")
