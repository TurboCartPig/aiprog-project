import tensorflow as tf
from tensorflow.keras import layers

import dataset

model = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(dataset.training, validation_data=dataset.validation, epochs=5)

# Evaluate the performance of the model, based on loss and accuracy
test_loss, test_acc = model.evaluate(dataset.validation, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
