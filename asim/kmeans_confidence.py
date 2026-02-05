"""
K-Means Color Confidence Module

This module provides a function to calculate a confidence score (0-1) based on
the color distribution within a predicted mask using K-means clustering.

The confidence score is the sum of top 4 colors' percentages (normalized to 0-1).
Higher score = more confidence = colors are more concentrated.
"""

import numpy as np
import cv2
from collections import Counter
from sklearn.cluster import KMeans
import torch


# ============================================
# K-MEANS HELPER FUNCTIONS
# ============================================

def crop_border_numpy(img_array, tolerance=20):
    """Crop black borders from image."""
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


def kmeans_cluster_numpy(img_array, k, border_color, tolerance=20):
    """Apply K-means clustering to image pixels."""
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
    """Generate K distinct colors using HSV color space."""
    colors = []
    for i in range(k):
        hue = int(180 * i / k)
        hsv = np.uint8([[[hue, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(rgb))
    return colors


def create_clustered_image(shape, labels, random_colors):
    """Create image with random colors for each cluster."""
    h, w = shape[:2]
    random_img = np.zeros((h*w, 3), dtype=np.uint8)
    for i in range(h*w):
        if labels[i] >= 0:
            random_img[i] = random_colors[labels[i]]
    return random_img.reshape(h, w, 3)


def fill_holes(img, box_sizes=[3, 5, 7, 11, 15]):
    """Fill black holes in image using neighborhood majority color."""
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
    """Restore processed image to original shape with black borders."""
    top, bottom, left, right = crop_bounds
    h, w = original_shape[:2]
    restored = np.zeros((h, w, 3), dtype=np.uint8)
    restored[top:bottom+1, left:right+1] = processed_img
    return restored


# ============================================
# MAIN FUNCTION: CALCULATE TOP-4 CONFIDENCE
# ============================================

def calculate_kmeans_confidence(image_tensor, mask_tensor, k=9, top_n=4):
    """
    Calculate confidence score based on K-means color distribution within mask.
    
    Args:
        image_tensor: Input image tensor [B, C, H, W] or [C, H, W] (normalized)
        mask_tensor: Predicted mask tensor [B, 1, H, W] or [1, H, W] or [H, W]
                     Values should be in range [0, 1] (after sigmoid)
        k: Number of K-means clusters (default: 9)
        top_n: Number of top colors to sum (default: 4)
    
    Returns:
        confidence: Float between 0 and 1 (sum of top-N colors' percentages / 100)
                    Higher = more concentrated colors in mask = more confidence
    """
    # Handle batched input - process only first image
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor[0]
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor[0]  # Remove channel dim
    
    # Convert to numpy
    # Denormalize image (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if image_tensor.device.type == 'cuda':
        mean = mean.cuda()
        std = std.cuda()
    
    img_denorm = (image_tensor * std + mean)
    img_np = (img_denorm.cpu().numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    # Threshold mask to binary (0.5 threshold)
    mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
    
    # Apply K-means processing
    cropped, border_color, crop_bounds = crop_border_numpy(img_np)
    labels, centers = kmeans_cluster_numpy(cropped, k, border_color)
    random_colors = generate_random_colors(k)
    kmeans_colored = create_clustered_image(cropped.shape, labels, random_colors)
    filled = fill_holes(kmeans_colored)
    restored = restore_to_original_shape(filled, img_np.shape, crop_bounds)
    
    # Resize mask to match if needed
    if mask_np.shape != restored.shape[:2]:
        mask_np = cv2.resize(mask_np, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Get pixels within mask
    white_mask = mask_np > 0
    if not np.any(white_mask):
        # No mask pixels - return low confidence
        return 0.0
    
    masked_pixels = restored[white_mask]
    
    # Calculate frequency of each color
    pixel_tuples = [tuple(p) for p in masked_pixels]
    freq_map = Counter(pixel_tuples)
    
    # Calculate percentages
    total_pixels = len(masked_pixels)
    frequencies = [(count / total_pixels * 100) for _, count in freq_map.most_common()]
    
    # Sum top N percentages (normalized to 0-1)
    top_n_sum = sum(frequencies[:top_n])
    confidence = top_n_sum / 100.0  # Normalize to 0-1
    
    return confidence


def apply_confidence_weighting(outputs, images, k=9, top_n=4):
    """
    Apply K-means confidence weighting to model outputs.
    
    Args:
        outputs: Model output logits tensor [B, 1, H, W]
        images: Input images tensor [B, C, H, W]
        k: Number of K-means clusters
        top_n: Number of top colors to consider
    
    Returns:
        weighted_outputs: outputs * confidence_score for each sample
    """
    batch_size = outputs.shape[0]
    weighted_outputs = outputs.clone()
    
    # Get probabilities for mask
    probs = torch.sigmoid(outputs)
    
    for b in range(batch_size):
        confidence = calculate_kmeans_confidence(
            images[b], 
            probs[b], 
            k=k, 
            top_n=top_n
        )
        weighted_outputs[b] = outputs[b] * confidence
    
    return weighted_outputs


# For quick testing
if __name__ == "__main__":
    print("K-Means Confidence Module Loaded!")
    print("Functions available:")
    print("  - calculate_kmeans_confidence(image_tensor, mask_tensor, k=9, top_n=4)")
    print("  - apply_confidence_weighting(outputs, images, k=9, top_n=4)")
