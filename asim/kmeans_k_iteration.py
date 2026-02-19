"""
K-Means Dissection: Iterate K from 3 to 9 on a single image
and save the results to a 'result' folder.

Usage:
    python kmeans_k_iteration.py                         # uses default image subi/1.png
    python kmeans_k_iteration.py --image path/to/img.png # uses a custom image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import argparse
from collections import Counter
from sklearn.cluster import KMeans


# ============================================
# FUNCTIONS (from kmeandisection.ipynb)
# ============================================

def load_image(path):
    img = Image.open(path)
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return img_array


def crop_border(img_array, tolerance=20):
    h, w = img_array.shape[:2]
    corners = [img_array[0, 0], img_array[0, w - 1],
               img_array[h - 1, 0], img_array[h - 1, w - 1]]
    border_color = np.mean(corners, axis=0).astype(np.uint8)

    def is_border(pixel):
        return np.all(np.abs(pixel.astype(int) - border_color.astype(int)) <= tolerance)

    def is_border_line(line, threshold=0.9):
        return sum(1 for p in line if is_border(p)) / len(line) > threshold

    top, bottom, left, right = 0, h - 1, 0, w - 1
    for i in range(h):
        if not is_border_line(img_array[i, :]):
            top = i
            break
    for i in range(h - 1, -1, -1):
        if not is_border_line(img_array[i, :]):
            bottom = i
            break
    for j in range(w):
        if not is_border_line(img_array[:, j]):
            left = j
            break
    for j in range(w - 1, -1, -1):
        if not is_border_line(img_array[:, j]):
            right = j
            break

    return img_array[top:bottom + 1, left:right + 1], border_color, (top, bottom, left, right)


def kmeans_cluster(img_array, k, border_color, tolerance=20):
    pixels = img_array.reshape(-1, 3)

    def is_border(p):
        return np.all(np.abs(p.astype(int) - border_color.astype(int)) <= tolerance)

    non_border_mask = np.array([not is_border(p) for p in pixels])
    non_border_pixels = pixels[non_border_mask]

    if len(non_border_pixels) == 0:
        return np.full(len(pixels), -1), np.zeros((k, 3), dtype=np.uint8)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(non_border_pixels)

    labels = np.full(len(pixels), -1)
    labels[non_border_mask] = kmeans.predict(non_border_pixels)

    return labels, kmeans.cluster_centers_.astype(np.uint8)


def generate_random_colors(k):
    colors = []
    for i in range(k):
        hue = int(180 * i / k)
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(rgb))
    return colors


def create_clustered_image(shape, labels, random_colors):
    h, w = shape[:2]
    random_img = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(h * w):
        if labels[i] >= 0:
            random_img[i] = random_colors[labels[i]]
    return random_img.reshape(h, w, 3)


def fill_holes(img, box_sizes=[3, 5, 7, 11, 15]):
    filled = img.copy()
    h, w = img.shape[:2]
    black_mask = np.all(filled <= 10, axis=2)
    positions = list(zip(*np.where(black_mask)))
    cx, cy = h // 2, w // 2
    positions.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    for i, j in positions:
        for box in box_sizes:
            half = box // 2
            t, b = max(0, i - half), min(h, i + half + 1)
            l, r = max(0, j - half), min(w, j + half + 1)
            neighborhood = filled[t:b, l:r].reshape(-1, 3)
            non_black = [tuple(p) for p in neighborhood if np.any(p > 10)]
            if non_black:
                filled[i, j] = Counter(non_black).most_common(1)[0][0]
                break
    return filled


def restore_to_original_shape(processed_img, original_shape, crop_bounds):
    top, bottom, left, right = crop_bounds
    h, w = original_shape[:2]
    restored = np.zeros((h, w, 3), dtype=np.uint8)
    restored[top:bottom + 1, left:right + 1] = processed_img
    return restored


def process_image_for_k(input_path, k):
    """Process a single image with a given K, return the result array."""
    original = load_image(input_path)
    cropped, border_color, crop_bounds = crop_border(original)
    labels, centers = kmeans_cluster(cropped, k, border_color)
    random_colors = generate_random_colors(k)
    kmeans_rand = create_clustered_image(cropped.shape, labels, random_colors)
    filled = fill_holes(kmeans_rand)
    restored = restore_to_original_shape(filled, original.shape, crop_bounds)
    return original, restored


# ============================================
# MAIN: iterate K from 3 to 9
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Run K-Means dissection on a single image for K=3..9 and save results."
    )
    parser.add_argument(
        "--image", type=str, default="subi/1.png",
        help="Path to the input image (default: subi/1.png)"
    )
    parser.add_argument(
        "--output", type=str, default="result",
        help="Output folder for results (default: result)"
    )
    args = parser.parse_args()

    input_path = args.image
    output_folder = args.output

    if not os.path.isfile(input_path):
        print(f"Error: image not found at '{input_path}'")
        return

    os.makedirs(output_folder, exist_ok=True)

    k_values = list(range(3, 10))  # K = 3, 4, 5, 6, 7, 8, 9
    results = {}

    print(f"Processing '{input_path}' with K = {k_values} ...")
    for k in k_values:
        print(f"  K={k} ...", end=" ", flush=True)
        original, restored = process_image_for_k(input_path, k)
        results[k] = restored

        # Save individual result
        out_path = os.path.join(output_folder, f"k{k}.png")
        Image.fromarray(restored).save(out_path)
        print(f"saved → {out_path}")

    # ------------------------------------------------------------------
    # Build a comparison figure: 2 rows
    #   Row 1: original image (spanning all columns)
    #   Row 2: K=3 .. K=9 side-by-side
    # ------------------------------------------------------------------
    n_k = len(k_values)
    fig, axes = plt.subplots(2, n_k, figsize=(4 * n_k, 8))

    # Top row: show the original in the middle, hide the rest
    mid = n_k // 2
    for idx in range(n_k):
        axes[0, idx].axis("off")
    axes[0, mid].imshow(original)
    axes[0, mid].set_title("Original", fontsize=14, fontweight="bold")

    # Bottom row: K-means results
    for idx, k in enumerate(k_values):
        axes[1, idx].imshow(results[k])
        axes[1, idx].set_title(f"K = {k}", fontsize=12)
        axes[1, idx].axis("off")

    plt.tight_layout()
    comparison_path = os.path.join(output_folder, "comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison figure saved → {comparison_path}")
    plt.show()


if __name__ == "__main__":
    main()
