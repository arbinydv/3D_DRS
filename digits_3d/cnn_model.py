import tensorflow as tf

def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 1), activation='relu'), # First convolution
        tf.keras.layers.MaxPooling2D((2, 1)),  # Pooling layer
        tf.keras.layers.Conv2D(32, (3, 1), activation='relu'),  # Second convolution
        tf.keras.layers.Flatten(),  # flatten
        tf.keras.layers.Dense(64, activation='relu'),  # connected layers during making dense
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# TODO : Build a custom optimizer
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model