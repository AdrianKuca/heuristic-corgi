from pathlib import Path
from statistics import mean
from matplotlib import pyplot as plt
import json


breed_ctr = {}
with open("dog_annotations.json", "r") as f:
    annotations = json.loads(f.read())
for dogs in annotations.values():
    for dog in dogs:
        if dog["breed"] in breed_ctr.keys():
            breed_ctr[dog["breed"]] += 1
        else:
            breed_ctr[dog["breed"]] = 1
print(breed_ctr)
breeds = []
counts = []
for key in sorted(breed_ctr, key=breed_ctr.get, reverse=True):
    breeds.append(key)
    counts.append(breed_ctr[key])

fig = plt.figure(figsize=(20, 10), tight_layout=True)
gs = fig.add_gridspec(4, 1)
ax = fig.add_subplot(gs[0:3, 0])
ax.bar(range(0, len(counts)), counts, align="center")
ax.set_title("Dogs per breed")
ax.set_xticks(range(0, len(counts)), breeds, rotation="vertical")
ax.axhline(
    y=max(counts),
    label="Maximum dog per breed count",
    linestyle=":",
    color="red",
)
ax.text(
    len(counts) / 2,
    max(counts) + 1,
    "Maximum dog per breed count: " + str(max(counts)),
    fontsize=20,
)
ax.axhline(
    y=min(counts),
    label="Minimum dog per breed count",
    linestyle=":",
    color="red",
)
ax.text(
    len(counts) / 2 + len(counts) / 4,
    min(counts) + 1,
    "Minimum dog per breed count: " + str(min(counts)),
    fontsize=20,
)
dir = Path(Path(__file__).parent, "dogs_per_breed")
dir.mkdir(parents=True, exist_ok=True)
plt.savefig(Path(dir, "Dogs_per_breed.png"))
plt.show()
