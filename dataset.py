"""
Module for loading and preprocessing the dataset, for further use in a specific model.
"""

import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Label discriminant to class name
classes = {
    1: "Danaus plexippus",
    2: "Heliconius charitonius",
    3: "Heliconius erato",
    4: "Junonia coenia",
    5: "Lycaena phlaeas",
    6: "Nymphalis antiopa",
    7: "Papilio cresphontes",
    8: "Pieris rapae",
    9: "Vanessa atalanta",
    10: "Vanessa cardui",
}

dataset_dir = "./datasets/gen"
labels = "inferred"
label_mode = "int"
color_mode = "rgb"
batch_size = 32
image_size = (256, 256)
seed = 1223334444
validation_split = 0.2
subset = "training"
interpolation = "lanczos5"

# Find the label for each image based on it's name
# dirpath, dirnames, filenames = next(os.walk(dataset_dir))
# labels = list(map(lambda name: int(name[:3]), filenames))

training = (
    image_dataset_from_directory(
        dataset_dir,
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation=interpolation,
    )
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation = (
    image_dataset_from_directory(
        dataset_dir,
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation=interpolation,
    )
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Show a sample of the training dataset to validate correct loading
    plt.figure(figsize=(10, 10))
    for images, labels in training.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(training.class_names[labels[i]])
            plt.axis("off")
    plt.show()
