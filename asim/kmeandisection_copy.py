# Import all required packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd

# Ensure output directories exist
os.makedirs('subcolr', exist_ok=True)
os.makedirs('subres', exist_ok=True)

# ============================================
# CORE FUNCTIONS
# ============================================

def load_image(path):
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return img_array

def crop_border(img_array, tolerance=20):
    h, w = img_array.shape[:2]
    corners = [img_array[0,0], img_array[0,w-1], img_array[h-1,0], img_array[h-1,w-1]]
    border_color = np.mean(corners, axis=0).astype(np.uint8)
    
    def is_border(pixel):
        return np.all(np.abs(pixel.astype(int) - border_color.astype(int)) <= tolerance)
    def is_border_line(line, threshold=0.9):
        return sum(1 for p in line if is_border(p)) / len(line) > threshold
    
    top, bottom, left, right = 0, h-1, 0, w-1
    for i in range(h):
        if not is_border_line(img_array[i, :]): top = i; break
    for i in range(h-1, -1, -1):
        if not is_border_line(img_array[i, :]): bottom = i; break
    for j in range(w):
        if not is_border_line(img_array[:, j]): left = j; break
    for j in range(w-1, -1, -1):
        if not is_border_line(img_array[:, j]): right = j; break
    
    return img_array[top:bottom+1, left:right+1], border_color, (top, bottom, left, right)

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
    # Consistent random colors for reproducibility if needed, or purely random
    # Using the HSV logic from original notebook for distinctness
    colors = []
    for i in range(k):
        hue = int(180 * i / k)
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(rgb))
    return colors

def create_clustered_image(shape, labels, random_colors):
    h, w = shape[:2]
    random_img = np.zeros((h*w, 3), dtype=np.uint8)
    for i in range(h*w):
        if labels[i] >= 0:
            random_img[i] = random_colors[labels[i]]
    return random_img.reshape(h, w, 3)

def fill_holes(img, box_sizes=[3, 5, 7, 11, 15]):
    filled = img.copy()
    h, w = img.shape[:2]
    black_mask = np.all(filled <= 10, axis=2)
    positions = list(zip(*np.where(black_mask)))
    cx, cy = h // 2, w // 2
    positions.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    for i, j in positions:
        for box in box_sizes:
            half = box // 2
            t, b = max(0, i-half), min(h, i+half+1)
            l, r = max(0, j-half), min(w, j+half+1)
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
    restored[top:bottom+1, left:right+1] = processed_img
    return restored

def process_single_image(input_path, output_path, k=10):
    try:
        original = load_image(input_path)
        if original is None:
            print(f"Failed to load {input_path}")
            return None
            
        cropped, border_color, crop_bounds = crop_border(original)
        labels, centers = kmeans_cluster(cropped, k, border_color)
        # Use fixed seed for generating colors if consistency across runs is desired, 
        # but here we generate per image as per original code
        random_colors = generate_random_colors(k)
        kmeans_rand = create_clustered_image(cropped.shape, labels, random_colors)
        filled = fill_holes(kmeans_rand)
        restored = restore_to_original_shape(filled, original.shape, crop_bounds)
        
        Image.fromarray(restored).save(output_path)
        return restored
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

# ============================================
# BATCH K-MEANS & FREQUENCY ANALYSIS
# ============================================

K_CLUSTERS = 7
INPUT_DIR = 'subi'
GT_DIR = 'subgt'
KMEANS_OUT_DIR = 'subcolr'
FINAL_RES_DIR = 'subres'
LOG_FILE = os.path.join(FINAL_RES_DIR, 'frequency_analysis_log.txt')

# Clear previous log if exists
with open(LOG_FILE, 'w') as f:
    f.write("Processing Log\n================\n")

# Get all images
# Use a more robust number extraction for sorting
def get_image_number(filename):
    try:
        return int(os.path.splitext(filename)[0])
    except ValueError:
        return float('inf') # Put non-numbered files at the end

all_images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')], 
                   key=get_image_number)

print(f"Found {len(all_images)} images to process.")

for i, img_name in enumerate(all_images):
    print(f"\nProcessing [{i+1}/{len(all_images)}]: {img_name}")
    
    # Paths
    input_path = os.path.join(INPUT_DIR, img_name)
    kmeans_path = os.path.join(KMEANS_OUT_DIR, img_name)
    gt_path = os.path.join(GT_DIR, img_name)
    plot_path = os.path.join(FINAL_RES_DIR, f"{img_name.split('.')[0]}_plot.png")
    
    # --- Step 1: Ensure K-Means Image Exists ---
    colored_img = None
    if os.path.exists(kmeans_path):
        print(f"  -> Loading existing K-means result from {kmeans_path}")
        colored_img = load_image(kmeans_path)
    else:
        print(f"  -> Generating K-means image (K={K_CLUSTERS})...")
        colored_img = process_single_image(input_path, kmeans_path, k=K_CLUSTERS)
    
    if colored_img is None:
        print("  -> Skipping analysis due to missing labeled image.")
        continue

    # --- Step 2: Frequency Analysis with Mask ---
    if not os.path.exists(gt_path):
        print(f"  -> Skipping frequency analysis: Mask not found at {gt_path}")
        continue
        
    gt_img = load_image(gt_path)
    if gt_img is None:
        print(f"  -> Failed to load mask: {gt_path}")
        continue

    # Create binary mask (White > 200)
    # Handle both grayscale (2D) and RGB (3D) masks
    if len(gt_img.shape) == 3:
        white_mask = np.all(gt_img > 200, axis=2)
    else:
        white_mask = gt_img > 200

    # Extract pixels under the mask
    # colored_img is RGB, shape (H, W, 3)
    # mask is boolean, shape (H, W)
    # Ensure dimensions match (simple check)
    if colored_img.shape[:2] != white_mask.shape[:2]:
        print(f"  -> Shape mismatch! Image: {colored_img.shape}, Mask: {white_mask.shape}. Resizing mask.")
        # Resize mask to match image if needed
        pil_mask = Image.fromarray(white_mask)
        pil_mask = pil_mask.resize((colored_img.shape[1], colored_img.shape[0]), resample=Image.NEAREST)
        white_mask = np.array(pil_mask)

    masked_pixels = colored_img[white_mask]
    
    if len(masked_pixels) == 0:
        print("  -> Mask is empty (no white pixels). Skipping stats.")
        continue

    # Count colors
    pixel_tuples = [tuple(p) for p in masked_pixels]
    freq_map = Counter(pixel_tuples)
    
    # Create DataFrame
    freq_df = pd.DataFrame(
        [(rgb, count) for rgb, count in freq_map.most_common()],
        columns=['RGB', 'Frequency']
    )
    freq_df['Percentage'] = (freq_df['Frequency'] / len(masked_pixels) * 100).round(2)
    
    # Calculate Top 3 Sum
    top3_sum = freq_df.head(3)['Percentage'].sum() if len(freq_df) >= 1 else 0.0

    # --- Step 3: Generate Plot & Log ---
    
    # 1. Visualization Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(colored_img)
    axes[0].set_title(f'{img_name} - Clustered (K={K_CLUSTERS})')
    axes[0].axis('off')
    
    axes[1].imshow(gt_img, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Visualize only masked region
    masked_view = np.zeros_like(colored_img)
    masked_view[white_mask] = colored_img[white_mask]
    axes[2].imshow(masked_view)
    axes[2].set_title('Masked Region Colors')
    axes[2].axis('off')
    
    # Color bar / text info
    top_colors = freq_df.head(10)
    color_bar = np.zeros((50, len(top_colors)*50 if len(top_colors) > 0 else 50, 3), dtype=np.uint8)
    if len(top_colors) > 0:
        for c_idx, (rgb, _, _) in enumerate(top_colors.values):
            color_bar[:, c_idx*50:(c_idx+1)*50] = rgb
    
    axes[3].imshow(color_bar)
    axes[3].set_title(f'Top {len(top_colors)} Colors')
    axes[3].axis('off')
    
    plt.suptitle(f"Image: {img_name} | Top 3 Sum: {top3_sum:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()  # Close figure to free memory
    
    # 2. Text Log
    log_text = f"\n=== Image {img_name} - Frequency Map ===\n"
    log_text += f"Total masked pixels: {len(masked_pixels)}\n"
    log_text += f"Unique colors: {len(freq_map)}\n"
    log_text += freq_df.head(10).to_string(index=False)
    log_text += f"\n>>> TOP 3 COLORS COMBINED: {top3_sum:.2f}%\n"
    log_text += "-" * 40
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_text)
        
    print(f"  -> Saved plot to {plot_path}")
    print(f"  -> Logged stats (Top 3: {top3_sum:.2f}%)")

print("\nBatch processing complete!")
