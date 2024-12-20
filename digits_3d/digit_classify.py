from classifer_model import predict_digit

# Entry point of the model
def digit_classify(sample_data):
    prediction = predict_digit(sample_data)
    return prediction[0]

if __name__ == '__main__':
    digit_classify("training_data/stroke_5_0017.csv")

