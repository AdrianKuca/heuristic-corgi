# heuristic-corgi
By using this dataset, one can probably get this repo to work:
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset?resource=download

Its supposed to work in VSCODE dev container (made with "Dockerfile" in WSL2 Ubuntu image [kinda complicated, I know...]), and the path mapping is in the "devcontainer.json"

After the dataset is unzipped, two directories show up: Annotations and Images.

Annotations: 
    Those have to be processed with "process_annotations.py" which outputs "dog_annotations.json".

Images:
    By using the bounding boxes and breeds contained in "dog_annotations.json", "process_images.py" outputs (cropped, grayscale, processed and labeled) images into test and training datasets.
    Output datasets can be configured in "datasets.py".

Training:
    "train.py" trains the model using the chosen dataset. Best test accuracy was around 98%.