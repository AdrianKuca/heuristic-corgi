# https://www.tensorflow.org/tutorials/images/cnn
from tensorflow.keras import layers, models, datasets

input_size = 30
model = models.Sequential()
model.add(
    layers.Conv2D(
        input_size, (3, 3), activation="relu", input_shape=(input_size, input_size, 1)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(input_size * 2, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(input_size * 2, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(input_size * 2, activation="relu"))
model.add(layers.Dense(10))
