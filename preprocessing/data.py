import os, re
from PIL import Image
from pathlib import Path
import numpy as np
from configuration.paths import TRAINING_PATH
from configuration.datasets import labels_in_dataset
import multiprocessing as mp


def worker(path):
    label = re.sub("_\d*\.jpg", "", path.name)
    dog = Image.open(path)
    pix = np.array(dog.getdata()).reshape(dog.size[0], dog.size[1], 1).astype(np.int8)
    return pix, label


def get_dataset(path, train_or_test):
    images_path = str(path) + "-images.npy"
    labels_path = str(path) + "-labels.npy"
    if not (Path(images_path).exists() and Path(labels_path).exists()):
        print("\tReading from disk")
        files = os.listdir(path / train_or_test)
        paths = [path / train_or_test / file for file in files]
        images_labels = mp.Pool(processes=16).map(worker, paths)

        images = np.array([x[0] for x in images_labels], dtype=np.int8)
        np.save(images_path, images)

        labels = np.array(
            [labels_in_dataset.index(x[1]) for x in images_labels], dtype=np.int32
        )
        np.save(labels_path, labels)
    else:
        print("\tReading from dump")
        images = np.load(images_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)

    return images.astype(np.int8), labels


if __name__ == "__main__":
    train_images, train_labels = get_dataset(TRAINING_PATH("dnq60x60"), "training")
