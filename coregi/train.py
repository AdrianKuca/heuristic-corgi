import json, os, shutil
from datetime import datetime
from model import get_model
from preprocessing.data import get_dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from configuration.paths import TRAINING_PATH, RESULTS_PATH

SIZE = 60
DATASET = f"ndnq{SIZE}x{SIZE}"
NOW = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

results_path = RESULTS_PATH(DATASET) / NOW
checkpoints_path = RESULTS_PATH(DATASET) / NOW / "checkpoints"

print("Getting training dataset")
train_images, train_labels = get_dataset(TRAINING_PATH(DATASET), "training")

print("Getting testing dataset")
test_images, test_labels = get_dataset(TRAINING_PATH(DATASET), "testing")


print("Compiling model")
model = get_model(SIZE)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoints_path / "{loss}", save_weights_only=True
)
if tf.train.latest_checkpoint(checkpoints_path) != None:
    if input(tf.train.latest_checkpoint(checkpoints_path) + " load? y/n") == "y":
        model.load_weights(tf.train.latest_checkpoint(checkpoints_path))

history = model.fit(
    train_images,
    train_labels,
    epochs=2,
    callbacks=[checkpoint_callback],
    validation_data=(test_images, test_labels),
)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

# Save history and model file for further analysys
history.history["test_loss"] = test_loss
history.history["test_acc"] = test_acc
jsoned_history = json.dumps(history.history, indent=4)


os.makedirs(results_path, exist_ok=True)
with open(results_path / "history.json", "w") as f:
    f.write(jsoned_history)

shutil.copyfile("./coregi/model.py", results_path / "model.py")
