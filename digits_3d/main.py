from digits_3d.model_building import predict_digit, train_digit_classifier

def main():
    print("Training the model...")
    train_digit_classifier()  # Trains the model

    # Test the model with a random stroke
    print("Testing the model...")
    predict_digit('single_test/stroke_0_0002.csv')  # Example test stroke

if __name__ == "__main__":
    main()
