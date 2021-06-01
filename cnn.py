import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, layers

import dataset

# Setup and define model
model = Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="selu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", data_format="channels_last"),
        tfa.layers.GroupNormalization(groups=32, axis=3),
        layers.Flatten(),
        layers.Dense(96, activation="relu"),
        layers.Dense(96, activation="sigmoid"),
        layers.Dense(10),
    ]
)

# Compile model
model.compile(
    optimizer="adamax",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fit model to dataset
model.fit(dataset.training, validation_data=dataset.validation, epochs=8)

if __name__ == "__main__":
    # Evaluate the performance of the model, based on loss and accuracy
    test_loss, test_acc = model.evaluate(dataset.validation, verbose=2)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(dataset.validation)
    pred_categories = tf.argmax(y_pred, axis=1)
    true_categories = tf.concat([y for x, y in dataset.validation], axis=0)
    print(confusion_matrix(pred_categories, true_categories))
