from digits_3d.model_building import train_digit_classifier, predict_digit

def digit_classify(sample, training_data='training_data'):
    print("Training the model...")
    train_digit_classifier(training_data)

    print("\nTesting the model with the given sample...")
    predict_digit(sample)

if __name__ == "__main__":
    # test the classifier with single stroke file
    digit_classify('single_test/stroke_0_0002.csv')
