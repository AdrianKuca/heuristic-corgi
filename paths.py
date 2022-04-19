from pathlib import Path


DATA_PATH = Path("/workspace/")
ANNOTATIONS_PATH = Path("/workspace/annotations/Annotation/")
IMAGES_PATH = Path("/workspace/images/Images/")
TRAINING_PATH = lambda i: Path(f"/workspace/training_{i}/")
