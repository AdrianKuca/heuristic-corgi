import matplotlib.pyplot as plt
import os, json
from pathlib import Path
import importlib.util


def get_model_info(model_module, size):
    model_info = []
    model = model_module.get_model(size)
    model.summary(print_fn=lambda x: model_info.append(x))
    return model_info


def plot_history(history, model_info):
    fig = plt.figure(figsize=(18, 10), tight_layout=True)
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    text_area = fig.add_subplot(gs[0, 3])
    text_area.axis("off")
    text_area.text(0, 0, "\n".join(model_info), fontsize=10, family="monospace")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    ax1.plot(history["accuracy"], label="Accuracy", color="red")
    ax1.set_ylim(0, 1)
    ax1.axhline(
        y=history["test_acc"], label="Test accuracy", linestyle=":", color="red"
    )
    ax1.text(
        0,
        history["test_acc"],
        "Test accuracy: " + history["test_acc"],
        fontsize=10,
        family="monospace",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    ax2.plot(history["loss"], label="Loss", color="blue")
    ax2.axhline(y=history["test_loss"], label="Test loss", linestyle=":", color="blue")
    ax2.text(
        0,
        history["test_loss"],
        "Test loss: " + history["test_loss"],
        fontsize=10,
        family="monospace",
    )

    ax1.legend(loc="center left")
    ax2.legend(loc="center right")
    dir = Path(Path(__file__).parent, "train_acc_loss")
    dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(dir, model_info[0].split('"')[1].split('"')[0] + ".png"))


if __name__ == "__main__":

    # Walk over all results
    for root, dirs, files in os.walk("D:\PythonoweLove\pieski"):
        for dir in dirs:
            if dir.startswith("results_"):
                for datedir in os.listdir(Path(root, dir)):
                    history_file = Path(root, dir, datedir, "history.json")
                    model_file = Path(root, dir, datedir, "model.py")
                    if history_file.exists() and model_file.exists():

                        # Model
                        spec = importlib.util.spec_from_file_location(
                            "model", str(model_file)
                        )
                        model = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(model)
                        model_info = get_model_info(
                            model, int(dir.split("_")[1].split("x")[1])
                        )

                        # History
                        history = json.loads(history_file.read_text("utf-8"))
                        plot_history(history, model_info)
