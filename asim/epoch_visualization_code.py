# ================================================================
# EPOCH VISUALIZATION CODE FOR polymix-kmean-asim.ipynb
# ================================================================
# Add these functions and modifications to your notebook
# ================================================================

# ----------------------------------------
# STEP 1: Add this NEW CELL after Block 8.5 (Precompute K-Means)
# Copy everything between the triple quotes below
# ----------------------------------------

"""
# Block 8.6: Epoch Visualization Function

def create_kmeans_colorized(img_np, k=9):
    '''
    Apply K-means clustering and return colorized image.
    Matches the visualization from kmeandisection.ipynb
    '''
    h, w = img_np.shape[:2]
    pixels = img_np.reshape(-1, 3)
    
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Generate distinct colors for each cluster (like kmeandisection.ipynb)
        np.random.seed(42)
        colors = []
        for i in range(k):
            # Use HSV color space for vibrant distinct colors
            hue = int(180 * i / k)
            hsv = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors.append(rgb)
        
        # Create colored image based on cluster labels
        colored = np.zeros_like(pixels)
        for i, label in enumerate(labels):
            colored[i] = colors[label]
        colored = colored.reshape(h, w, 3)
        return colored
    except Exception as e:
        print(f"K-means colorization error: {e}")
        return img_np  # Return original on error


def visualize_epoch_progress(model, val_dataset, epoch, device, output_dir, k=9):
    '''
    Visualize model progress on a random validation image at end of epoch.
    Shows: Original Image | Model Prediction | Ground Truth | K-Means Colorized
    '''
    model.eval()
    
    # Pick a random validation sample
    rand_idx = random.randint(0, len(val_dataset) - 1)
    image, mask, filename = val_dataset[rand_idx]
    
    # Add batch dimension and move to device
    image_batch = image.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_batch)
        pred = (torch.sigmoid(output) > 0.5).float()
    
    # De-normalize image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_denorm = (image.cpu() * std) + mean
    img_np = img_denorm.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np_uint8 = (img_np * 255).astype(np.uint8)
    
    gt_mask = mask.cpu().squeeze().numpy()
    pred_mask = pred[0].cpu().squeeze().numpy()
    
    # Create K-means colorized version (like kmeandisection.ipynb)
    kmeans_colored = create_kmeans_colorized(img_np_uint8, k=k)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original: {filename[:20]}...' if len(filename) > 20 else f'Original: {filename}')
    axes[0].axis('off')
    
    # Model Prediction
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Model Prediction')
    axes[1].axis('off')
    
    # Ground Truth
    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # K-Means Colorized (like kmeandisection.ipynb)
    axes[3].imshow(kmeans_colored)
    axes[3].set_title(f'K-Means Colorized (K={k})')
    axes[3].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Validation Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f'epoch_{epoch:02d}_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"  📊 Epoch {epoch} visualization saved: {save_path}")


print("✅ Epoch visualization function defined!")
"""

# ----------------------------------------
# STEP 2: MODIFY the Training Loop (Block 12)
# Find this line in the training loop (after validation):
#     print(f"  No improvement for {epochs_without_improvement} epoch(s)")
# 
# ADD THIS LINE RIGHT AFTER IT:
# ----------------------------------------

"""
    # ========== EPOCH VISUALIZATION (After each epoch) ==========
    # Visualize random validation image with prediction and K-means colorization
    visualize_epoch_progress(model, val_dataset, epoch + 1, DEVICE, OUTPUT_DIR, k=9)
    # ============================================================
"""

# ----------------------------------------
# COMPLETE MODIFIED TRAINING LOOP - Replace Block 12 with this:
# ----------------------------------------

"""
# Block 12: Training Loop with Early Stopping + Epoch Visualization

print("="*60)
print(f"Starting Training with PolypMix Augmentation")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Model selection based on: Dice Score")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f">>> TRAINING ON: {DEVICE} <<<")
print("="*60)

train_losses = []
val_losses = []
train_dices = []
val_dices = []
train_ious = []
val_ious = []

# Best model tracking (based on Dice score)
best_dice = 0.0
best_epoch = 0
epochs_without_improvement = 0

training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 40)
    
    # Reshuffle augmentation pairs at start of each epoch
    if USE_POLYPMIX and hasattr(train_dataset, 'reshuffle_pairs'):
        train_dataset.reshuffle_pairs()
    
    # Training
    train_loss, train_iou, train_dice = train_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )
    
    # Validation
    val_loss, val_iou, val_dice = validate(
        model, val_loader, criterion, DEVICE
    )
    
    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)
    train_ious.append(train_iou)
    val_ious.append(val_iou)
    
    epoch_time = time.time() - epoch_start_time
    
    # Print results
    print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
    print(f"Epoch Time: {epoch_time:.1f}s")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    
    # Save best model based on Dice score
    if val_dice > best_dice:
        best_dice = val_dice
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'best_iou': val_iou,
        }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  -> New best model saved! (Dice: {best_dice:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement for {epochs_without_improvement} epoch(s)")
    
    # ========== EPOCH VISUALIZATION ==========
    # Show random validation image + prediction + K-means colorized
    visualize_epoch_progress(model, val_dataset, epoch + 1, DEVICE, OUTPUT_DIR, k=9)
    # ==========================================
    
    # Early stopping check
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\\n[!] Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
        break

total_training_time = time.time() - training_start_time

print("\\n" + "="*60)
print(f"Training Complete!")
print(f"Best Validation Dice: {best_dice:.4f} (Epoch {best_epoch})")
print(f"Total epochs trained: {epoch + 1}")
print(f"Total training time: {total_training_time/60:.1f} minutes")
print("="*60)
"""

# ================================================================
# SUMMARY OF CHANGES:
# ================================================================
# 1. Added create_kmeans_colorized() function - applies K-means and colors clusters
# 2. Added visualize_epoch_progress() function - creates 4-panel visualization:
#    - Panel 1: Original validation image
#    - Panel 2: Model's prediction (current epoch)
#    - Panel 3: Ground truth mask
#    - Panel 4: K-means colorized version (like kmeandisection.ipynb)
# 3. Modified training loop to call visualize_epoch_progress() after each epoch
# 4. Saves visualization as epoch_XX_visualization.png in OUTPUT_DIR
# ================================================================
