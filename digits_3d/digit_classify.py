from digits_3d.model_building import train_digit_classifier, predict_digit

def digit_classify(sample):
    predict_digit(sample)

if __name__ == "__main__":
    digit_classify('single_test/stroke_0_0002.csv')
