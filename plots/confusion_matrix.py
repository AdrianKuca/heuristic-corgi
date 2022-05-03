import importlib
import os
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt
from plots.utils import get_model_info, get_model_name, load_model_module
from preprocessing.data import get_dataset
from configuration.paths import DATA_PATH, TRAINING_PATH, RESULTS_PATH
import multiprocessing as mp
import subprocess


def load_dataset(dataset):
    print("Getting testing dataset")
    test_images, test_labels = get_dataset(TRAINING_PATH(dataset), "testing")
    return test_images, test_labels


def worker(model_file, input_size, checkpoints_path, test_images, responses):
    print("Compiling model")

    spec = importlib.util.spec_from_file_location("model", str(model_file))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_model(input_size)
    model.summary()
    if tf.train.latest_checkpoint(checkpoints_path) != None:
        print("loading...")
        model.load_weights(tf.train.latest_checkpoint(checkpoints_path))
        print("running...")
        for image in test_images:
            responses.append(model(image))
        print("responsed!")


def run_model(model_file, input_size, checkpoints_path, test_images):
    manager = mp.Manager()
    responses = manager.list()
    process = mp.Process(
        target=worker,
        args=[model_file, input_size, checkpoints_path, test_images, responses],
    )
    process.start()
    process.join()
    process.close()
    if len(responses):
        return responses[0]


def plot_confusion_matrix(labels, predictions, model_name):
    confusion = tf.math.confusion_matrix(
        labels=labels, predictions=predictions, num_classes=120
    )

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title
    ax1.hist(confusion, bins=120)
    ax1.legend(loc="center left")
    dir = Path(Path(__file__).parent, "confusion_matrix")
    dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(dir, model_name + ".png"))


if __name__ == "__main__":

    # Walk over all results
    for root, dirs, files in os.walk(DATA_PATH):
        for dir in dirs:
            if dir.startswith("results_"):
                for datedir in os.listdir(Path(root, dir)):
                    history_file = Path(root, dir, datedir, "history.json")
                    model_file = Path(root, dir, datedir, "model.py")
                    dataset_name = dir.split("_")[1]
                    input_size = int(dataset_name.split("x")[1])
                    checkpoints_path = Path(root, dir, datedir, "checkpoints")
                    if (
                        checkpoints_path.exists()
                        and history_file.exists()
                        and model_file.exists()
                    ):
                        test_images, test_labels = load_dataset(dataset_name)
                        responses = run_model(
                            model_file, input_size, checkpoints_path, test_images
                        )
                        if responses:
                            model_module = load_model_module(model_file)
                            plot_confusion_matrix(
                                test_labels,
                                responses,
                                get_model_name(model_module, input_size),
                            )
                        else:
                            print("Not drawing...")
