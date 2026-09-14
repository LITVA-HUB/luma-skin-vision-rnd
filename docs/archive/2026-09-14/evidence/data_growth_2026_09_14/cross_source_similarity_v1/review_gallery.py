"""Display four highest-correlation candidate image pairs for content review."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
result = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
rows = json.loads((ROOT / "rows.json").read_text(encoding="utf-8"))
pairs = sorted(result["outcome"]["candidate_views"], key=lambda p: -p["gray_correlation"])[:4]
arrays = {}


def pixels(row):
    key = row["source"], row["role"]
    if key not in arrays:
        if row["source"] == "lapa":
            split = "val" if row["role"] == "validation" else "train"
            path = DATA / "lapa/prepared_192" / f"{split}_rgb.npy"
        else:
            path = DATA / "celeba_mask_hq/prepared_192_v1/images.npy"
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays[key][row["global_index"]]


fig, axes = plt.subplots(4, 2, figsize=(7, 12), dpi=140)
for index, pair in enumerate(pairs):
    for column, side in enumerate(("left", "right")):
        row = rows[pair[side]]
        rgb = pixels(row)
        if side == "right" and pair["horizontal_flip"]:
            rgb = rgb[:, ::-1]
        axes[index, column].imshow(rgb)
        axes[index, column].axis("off")
        axes[index, column].set_title(f"{row['source']} / {row['role']}\nPair {index+1}: corr={pair['gray_correlation']:.4f}, MAE={pair['rgb_mae']:.2f}", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "closest_four.png")
plt.close(fig)
(ROOT / "closest_four.json").write_text(json.dumps(pairs, indent=2)+"\n", encoding="utf-8")
print(ROOT / "closest_four.png")
