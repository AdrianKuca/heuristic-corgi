from tensorflow.keras import layers, models
from configuration.datasets import labels_in_dataset

filter_size = (5, 5)
pool_size = (2, 2)


def get_model(input_size):
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            input_size,
            filter_size,
            activation="relu",
            input_shape=(input_size, input_size, 1),
        )
    )
    model.add(layers.Conv2D(input_size * 2, filter_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Conv2D(input_size * 2, filter_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Conv2D(input_size * 2, filter_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Conv2D(input_size * 2, filter_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Conv2D(input_size * 2, filter_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size))
    model.add(layers.Flatten())
    model.add(layers.Dense(input_size * 2, activation="relu"))
    model.add(layers.Dense(len(labels_in_dataset)))
    return model
