# Processes all the images in the dataset, to a version with only edges, no colors and the same resolution.
import json, os

from paths import IMAGES_PATH, DATA_PATH, TRAINING_PATH
from PIL import Image, ImageFilter

MAX_SIZE = 60
PIXELS = ["#", "O", "/", "-", " "]
try:
    os.mkdir(TRAINING_PATH(1))
    os.mkdir(TRAINING_PATH(2))
    os.mkdir(TRAINING_PATH(3))
except:
    pass


def process(image, double_filter=True, quantize=False):
    ratio = min(MAX_SIZE / image.size[0], MAX_SIZE / image.size[1])
    image.thumbnail((image.size[0] * ratio, image.size[1] * ratio))
    image = image.filter(ImageFilter.FIND_EDGES)
    if double_filter:
        image = image.filter(ImageFilter.FIND_EDGES)
    if quantize:
        (b,) = image.split()
        b = b.point(lambda i: i // 5 * 5)
        image = Image.merge("L", (b,))
    return image


def print_as_ascii(image):
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            for i, pixel in enumerate(PIXELS):
                value = image.getpixel((x, y))
                threshold = 255 * ((len(PIXELS) - i - 2) / len(PIXELS))
                if value > threshold:
                    print(pixel * 2, end="")
                    break
        print("")


with open("dog_annotations.json", "r") as f:
    annotations = json.loads(f.read())
ctr = 0
for key, dogs in annotations.items():
    with Image.open(IMAGES_PATH / (key + ".jpg")) as im:
        for dog in dogs:
            grayscale = im.convert("L")
            cropped = grayscale.crop(
                (int(dog["xmin"]), int(dog["ymin"]), int(dog["xmax"]), int(dog["ymax"]))
            )
            # print_as_ascii(process(cropped))
            # input()
            processed = process(cropped)
            processed.save(TRAINING_PATH(1) / (dog["breed"] + "_" + str(ctr) + ".jpg"))
            ctr += 1
