import os
import tensorflow as tf
from pathlib import Path
from coregi.model import get_model
from preprocessing.data import get_dataset
from configuration.paths import DATA_PATH, TRAINING_PATH, RESULTS_PATH

checkpoints_path = RESULTS_PATH(DATASET) / "checkpoints"


def load_dataset(input_size):
    print("Getting testing dataset")
    BATCH_SIZE = 64
    test_images, test_labels = get_dataset(TRAINING_PATH(DATASET), "testing")
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(
        BATCH_SIZE
    )


def load_model(input_size):
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
            print("loading...")
            model.load_weights(tf.train.latest_checkpoint(checkpoints_path))


if __name__ == "__main__":

    # Walk over all results
    for root, dirs, files in os.walk(DATA_PATH):
        for dir in dirs:
            if dir.startswith("results_"):
                for datedir in os.listdir(Path(root, dir)):
                    history_file = Path(root, dir, datedir, "history.json")
                    model_file = Path(root, dir, datedir, "model.py")
                    input_size = int(dir.split("_")[1].split("x")[1])
