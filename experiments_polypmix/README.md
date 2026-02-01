# PolypMix Augmentation for Polyp Segmentation

A Kaggle-ready notebook implementing **PolypMix** - a polyp-aware data augmentation technique for semi-supervised polyp segmentation.

## 📋 Overview

This implementation combines:
- **Swin-UNet** architecture for polyp segmentation
- **PolypMix augmentation** for data-efficient training
- **Early stopping** with Dice score-based model selection

## 🔬 How PolypMix Works

PolypMix is a **polyp-aware image mixing** technique that intelligently combines two training images:

```
mix_factor = mask1 / (mask0 + mask1)
mixed_image = image0 × (1 - mix_factor) + image1 × mix_factor
```

**Key Insight:** The mixing favors polyp regions over background, creating synthetic training samples that preserve polyp information.

### Visual Explanation

```
Image A (with polyp)     Image B (with polyp)
        ↓                        ↓
    [Mask A]                 [Mask B]
        ↓                        ↓
        └──────── mix_factor = B / (A + B) ────────┘
                         ↓
              [Mixed Image + Pseudo Label]
```

## 📁 Notebook Structure

| Block | Description |
|-------|-------------|
| **1** | Install dependencies, import libraries, set random seed |
| **2** | Define `SimplePolypMixAugmentor` and `PolypMixDataset` classes |
| **3** | Configure data paths for Kaggle/Colab/local environments |
| **4** | Set training hyperparameters and early stopping patience |
| **5** | Define Swin-UNet model architecture |
| **6** | Define loss functions (BCE + Dice) and metrics |
| **7** | Define `PolypDataset` class for loading images/masks |
| **8** | Create data loaders with optional PolypMix augmentation |
| **9** | Visualize augmented samples |
| **10** | Initialize model, optimizer |
| **11** | Define training and validation functions |
| **12** | Training loop with early stopping (Dice-based) |
| **13** | Plot training results (Loss, Dice, IoU) |
| **14** | Visualize sample predictions |

## ⚙️ Configuration

Edit **Block 4** to customize training:

```python
# Training parameters
TRAINING_SIZE = 25           # Labeled samples: 25, 50, 75, or 100
NUM_EPOCHS = 50              # Max epochs (early stopping may trigger sooner)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# PolypMix parameters
USE_POLYPMIX = True          # Enable/disable augmentation
POLYPMIX_RATIO = 1.0         # 1.0 = doubles the dataset
POLYPMIX_THRESHOLD = 0.7     # Confidence threshold for mixing

# Early stopping
EARLY_STOPPING_PATIENCE = 5  # Stop after N epochs without improvement
```

## 🚀 Quick Start (Kaggle)

1. **Upload** `Capstone_PolypMix_Kaggle.ipynb` to Kaggle
2. **Add dataset** as input (e.g., CVC-ClinicDB-612)
3. **Update path** in Block 3:
   ```python
   DATA_ROOT = "/kaggle/input/your-dataset-name/CVC-ClinicDB-612"
   ```
4. **Run all cells** - outputs save to `/kaggle/working/`

## 📊 Key Components

### 1. SimplePolypMixAugmentor

Mixes two image-mask pairs using ground truth masks:

```python
augmentor = SimplePolypMixAugmentor(threshold=0.7)
mixed_img, mixed_mask = augmentor(img0, mask0, img1, mask1)
```

### 2. PolypMixDataset

Wraps a base dataset to add augmented samples:

```python
augmented_dataset = PolypMixDataset(
    base_dataset=train_dataset,
    augment_ratio=1.0,    # Doubles the dataset
    threshold=0.7
)
# Original: 25 samples → Augmented: 50 samples (25 original + 25 mixed)
```

### 3. Early Stopping

Training stops when validation Dice score doesn't improve:

```python
if val_dice > best_dice:
    best_dice = val_dice
    save_model()
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
    break  # Stop training
```

## 📈 Expected Outputs

After training, find these in `OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (by Dice score) |
| `training_results.png` | Loss, Dice, IoU curves |
| `sample_predictions.png` | Visual comparison of predictions |
| `polypmix_samples.png` | Visualization of augmented training data |

## 🔧 Model Architecture

**Swin-UNet**: Combines Swin Transformer encoder with CNN decoder

```
Input (224×224×3)
    ↓
[Swin Transformer Encoder] ← Pretrained (microsoft/swin-base-patch4-window7-224)
    ↓
    s1 (56×56, 128ch) ──────────────────┐
    s2 (28×28, 256ch) ─────────────┐    │
    s3 (14×14, 512ch) ────────┐    │    │
    s4 (7×7, 1024ch)          │    │    │
    ↓                         │    │    │
[Decoder with Skip Connections]   │    │
    d4 ← s3 ─────────────────────┘    │
    d3 ← s2 ──────────────────────────┘
    d2 ← s1 ───────────────────────────┘
    ↓
Output (224×224×1) - Segmentation mask
```

## 📚 Reference

This implementation is based on:

> Jia et al., "PolypMixNet: Enhancing Semi-Supervised Polyp Segmentation with Polyp-Aware Augmentation", Computers in Biology and Medicine, 2024

## 📝 License

See repository license.