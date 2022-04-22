from model import model
from preprocessing.data import get_dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from configuration.paths import TRAINING_PATH

print("Getting training dataset")
train_images, train_labels = get_dataset(TRAINING_PATH("dnq30x30"), "training")

print("Getting testing dataset")
test_images, test_labels = get_dataset(TRAINING_PATH("dnq30x30"), "testing")

print("Compiling model")
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)
)

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
