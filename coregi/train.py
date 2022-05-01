import json, os, shutil
from datetime import datetime
from coregi.model import get_model
from preprocessing.data import get_dataset
from matplotlib import pyplot as plt
import tensorflow as tf
from configuration.paths import TRAINING_PATH, RESULTS_PATH

EPOCHS = 100
SIZE = 60
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 20000
DATASET = f"nf{SIZE}x{SIZE}"
NOW = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

results_path = RESULTS_PATH(DATASET) / NOW

print("Getting training dataset")
train_images, train_labels = get_dataset(TRAINING_PATH(DATASET), "training")
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

print("Getting testing dataset")
test_images, test_labels = get_dataset(TRAINING_PATH(DATASET), "testing")
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
    BATCH_SIZE
)


print("Compiling model")
model = get_model(SIZE)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

os.makedirs(results_path, exist_ok=True)
shutil.copyfile("./coregi/model.py", results_path / "model.py")
checkpoints_path = results_path / "checkpoints"


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoints_path / "{loss}", save_weights_only=True
)
if tf.train.latest_checkpoint(checkpoints_path) != None:
    if input(tf.train.latest_checkpoint(checkpoints_path) + " load? y/n") == "y":
        print("loading...")
        model.load_weights(tf.train.latest_checkpoint(checkpoints_path))

dataset_batch = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
history = model.fit(
    dataset_batch,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback],
)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(test_acc)

# Save history and model file for further analysys
history.history["test_loss"] = test_loss
history.history["test_acc"] = test_acc
jsoned_history = json.dumps(history.history, indent=4)

os.makedirs(results_path, exist_ok=True)
with open(results_path / "history.json", "w") as f:
    f.write(jsoned_history)
