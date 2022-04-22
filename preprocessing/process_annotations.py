# Processes annotations into single dogs json.
import os, json
import xml.etree.ElementTree as ET
from configuration.paths import ANNOTATIONS_PATH


def get_single_dogs(annotation_path):
    tree = ET.parse(annotation_path.as_posix())
    single_dogs = []
    for dogger in tree.findall("object"):
        annotation_data = {}
        annotation_data["breed"] = dogger.findtext("name")
        annotation_data["xmin"] = dogger.find("bndbox").findtext("xmin")
        annotation_data["ymin"] = dogger.find("bndbox").findtext("ymin")
        annotation_data["xmax"] = dogger.find("bndbox").findtext("xmax")
        annotation_data["ymax"] = dogger.find("bndbox").findtext("ymax")
        single_dogs.append(annotation_data)
    return single_dogs


dog_annotations = {}

# Get annotations as arrays of dicts where key is the fully qualifying doggers path
for root, dirs, files in os.walk(ANNOTATIONS_PATH):
    for breed_dir in dirs:
        breed_dir_path = ANNOTATIONS_PATH / breed_dir
        for root, dirs, files in os.walk(breed_dir_path):
            for annotation_file in files:
                annotation_path = breed_dir_path / annotation_file
                dog_annotations[f"{breed_dir}/{annotation_file}"] = get_single_dogs(
                    annotation_path
                )

with open("dog_annotations.json", "w") as f:
    f.write(json.dumps(dog_annotations))
