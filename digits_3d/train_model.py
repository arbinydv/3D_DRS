from digits_3d.cnn_model import build_cnn_model
from digits_3d.stroke_dataset import StrokeDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Step 1: Load and split data

    data_loader = StrokeDataset(data_dir="training_data", n_resample=16)
    X, y = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Step 2: Build the CNN model
    cnn_model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))

    # Step 3: Train the model
    history = cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Step 4: Evaluate the model
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2%}")

    # Step 5: Save the model
    cnn_model.save("stroke_digit_cnn.keras")

    # Step 6: Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()