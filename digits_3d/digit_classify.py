from digits_3d.model_building import predict_digit, digit_classify

def main():
    print("Training the model...")
    digit_classify('training_data')  # Trains the model

    # Test the model with a random stroke
    print("Testing the model...")
    predict_digit('single_test/stroke_0_0002.csv')  # Example test stroke

if __name__ == "__main__":
    main()
