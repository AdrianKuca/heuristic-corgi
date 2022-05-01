from pathlib import Path
import os

if os.name == "posix":
    DATA_PATH = Path("/workspace/")
    ANNOTATIONS_PATH = Path(DATA_PATH, "annotations/Annotation/")
    IMAGES_PATH = Path(DATA_PATH, "images/Images/")
    TRAINING_PATH = lambda i: Path(DATA_PATH, f"training_{i}/")
    CHECKPOINTS_PATH = lambda i: Path(DATA_PATH, f"checkpoints_{i}/")
    RESULTS_PATH = lambda i: Path(DATA_PATH, f"results_{i}/")
elif os.name == "nt":
    DATA_PATH = Path("D:\PythonoweLove\pieski")
    ANNOTATIONS_PATH = Path(DATA_PATH, "annotations/Annotation/")
    IMAGES_PATH = Path(DATA_PATH, "images/Images/")
    TRAINING_PATH = lambda i: Path(DATA_PATH, f"training_{i}/")
    CHECKPOINTS_PATH = lambda i: Path(DATA_PATH, f"checkpoints_{i}/")
    RESULTS_PATH = lambda i: Path(DATA_PATH, f"results_{i}/")
