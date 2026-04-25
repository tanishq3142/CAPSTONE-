"""
converttoplot.py
----------------
Reads all results_kmean_polypmix_ts_*.txt files in the current directory,
parses per-epoch training and validation metrics, and saves one figure per
file (Train Loss, Val Loss, Train Dice, Val Dice, Train IoU, Val IoU) plus
a combined comparison plot of Val Loss and Val Dice across all training sizes.

Output goes to:   ./outputs/
"""

import os
import re
import glob
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── regex to pull one epoch line ───────────────────────────────────────────
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+):\s+"
    r"Train Loss=([\d.]+),\s*Train Dice=([\d.]+),\s*Train IoU=([\d.]+)\s*\|\s*"
    r"Val Loss=([\d.]+),\s*Val Dice=([\d.]+),\s*Val IoU=([\d.]+)"
)

# ── colour palette (one per training-size file) ────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.framealpha": 0.85,
})


def parse_file(path):
    """Return a dict with lists: epochs, train_loss, val_loss, etc."""
    data = dict(epochs=[], train_loss=[], val_loss=[],
                train_dice=[], val_dice=[], train_iou=[], val_iou=[])
    meta = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # header key-value pairs
            if ":" in line and not line.startswith("Epoch"):
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()

            m = EPOCH_RE.match(line)
            if m:
                data["epochs"].append(int(m.group(1)))
                data["train_loss"].append(float(m.group(2)))
                data["train_dice"].append(float(m.group(3)))
                data["train_iou"].append( float(m.group(4)))
                data["val_loss"].append(  float(m.group(5)))
                data["val_dice"].append(  float(m.group(6)))
                data["val_iou"].append(   float(m.group(7)))

    return data, meta


def plot_single(data, meta, ts, color, out_path):
    """Three-panel figure: Loss | Dice | IoU for one training-size run."""
    epochs = data["epochs"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"PolypMix + K-Means   |   Training Size = {ts}",
        fontsize=14, fontweight="bold", y=1.01
    )

    panels = [
        ("Loss",  "train_loss", "val_loss"),
        ("Dice",  "train_dice", "val_dice"),
        ("IoU",   "train_iou",  "val_iou"),
    ]

    for ax, (metric, tr_key, vl_key) in zip(axes, panels):
        ax.plot(epochs, data[tr_key], label=f"Train {metric}",
                color=color, linewidth=1.8, linestyle="--", alpha=0.75)
        ax.plot(epochs, data[vl_key], label=f"Val {metric}",
                color=color, linewidth=2.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()

    # annotate best val dice
    best_dice = max(data["val_dice"])
    best_ep   = data["epochs"][data["val_dice"].index(best_dice)]
    axes[1].annotate(
        f"Best: {best_dice:.4f}\n(ep {best_ep})",
        xy=(best_ep, best_dice),
        xytext=(best_ep + max(1, len(epochs)//10), best_dice - 0.04),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=8, color="black"
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_comparison(all_data, out_path):
    """Two-panel figure: Val Loss & Val Dice for all training sizes overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Validation Curves — All Training Sizes (AUG-1)",
                 fontsize=14, fontweight="bold")

    for i, (ts, data, color) in enumerate(all_data):
        lbl = f"TS={ts}"
        ax1.plot(data["epochs"], data["val_loss"],  label=lbl,
                 color=color, linewidth=2)
        ax2.plot(data["epochs"], data["val_dice"],  label=lbl,
                 color=color, linewidth=2)

    for ax, title, ylabel in [
        (ax1, "Validation Loss",      "Loss"),
        (ax2, "Validation Dice Score","Dice"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(title="Training Size", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_iou_comparison(all_data, out_path):
    """Val IoU for all training sizes overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Validation IoU — All Training Sizes (AUG-1)",
                 fontsize=14, fontweight="bold")

    for ts, data, color in all_data:
        ax.plot(data["epochs"], data["val_iou"], label=f"TS={ts}",
                color=color, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(title="Training Size", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    txt_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "results_kmean_polypmix_ts_*.txt")))

    if not txt_files:
        print("No matching .txt files found in", SCRIPT_DIR)
        return

    print(f"Found {len(txt_files)} result file(s):\n")
    all_data = []   # list of (ts_label, data_dict, color)

    for i, path in enumerate(txt_files):
        fname = os.path.basename(path)
        # extract training size from filename, e.g. ts_25 → "25"
        m = re.search(r"ts_(\d+)", fname)
        ts    = m.group(1) if m else str(i)
        color = PALETTE[i % len(PALETTE)]

        print(f"Parsing: {fname}")
        data, meta = parse_file(path)

        if not data["epochs"]:
            print(f"  ⚠  No epoch data found – skipping.")
            continue

        # individual figure
        out_name = f"curves_ts_{ts}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plot_single(data, meta, ts, color, out_path)

        all_data.append((ts, data, color))

    if len(all_data) > 1:
        print("\nGenerating comparison plots …")
        plot_comparison(all_data,
                        os.path.join(OUTPUT_DIR, "comparison_val_loss_dice.png"))
        plot_iou_comparison(all_data,
                            os.path.join(OUTPUT_DIR, "comparison_val_iou.png"))

    print(f"\nDone. All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
