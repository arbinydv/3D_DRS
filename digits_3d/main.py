from digits_3d.model_building import predict_digit, train_digit_classifier

def main():
    print("Training the model...")
    train_digit_classifier()

    # Tests  the model
    print("Testing the model...")
    predict_digit('single_test/stroke_9_0052.csv')  # Example test stroke


if __name__ == "__main__":
    main()