"""
PolyMix Algorithm Comparison Plots
Compares AUG_RATIO_1 (1.0) vs AUG_RATIO_2 (2.0) across different training sizes.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
AUG_RATIO_1_DIR = BASE_DIR / "AUG_RATIO_1"
AUG_RATIO_2_DIR = BASE_DIR / "AUG_RATIO_2"
OUTPUT_DIR = BASE_DIR / "comparison_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Training sizes
TRAINING_SIZES = [25, 50, 75, 100, 490]

# Color schemes
COLORS = {
    1.0: '#2E86AB',  # Blue for ratio 1.0
    2.0: '#E94F37',  # Red for ratio 2.0
}

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10


def parse_result_file(filepath):
    """Parse a result file and extract epoch-wise metrics."""
    epochs = []
    train_loss, train_dice, train_iou = [], [], []
    val_loss, val_dice, val_iou = [], [], []
    best_dice, best_epoch = None, None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract epoch data
    epoch_pattern = r'Epoch (\d+): Train Loss=([\d.]+), Train Dice=([\d.]+), Train IoU=([\d.]+) \| Val Loss=([\d.]+), Val Dice=([\d.]+), Val IoU=([\d.]+)'
    for match in re.finditer(epoch_pattern, content):
        epochs.append(int(match.group(1)))
        train_loss.append(float(match.group(2)))
        train_dice.append(float(match.group(3)))
        train_iou.append(float(match.group(4)))
        val_loss.append(float(match.group(5)))
        val_dice.append(float(match.group(6)))
        val_iou.append(float(match.group(7)))
    
    # Extract best results
    best_pattern = r'Best Validation Dice: ([\d.]+)'
    best_epoch_pattern = r'Best Epoch: (\d+)'
    
    best_match = re.search(best_pattern, content)
    best_epoch_match = re.search(best_epoch_pattern, content)
    
    if best_match:
        best_dice = float(best_match.group(1))
    if best_epoch_match:
        best_epoch = int(best_epoch_match.group(1))
    
    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_loss),
        'train_dice': np.array(train_dice),
        'train_iou': np.array(train_iou),
        'val_loss': np.array(val_loss),
        'val_dice': np.array(val_dice),
        'val_iou': np.array(val_iou),
        'best_dice': best_dice,
        'best_epoch': best_epoch
    }


def load_all_data():
    """Load all experiment data."""
    data = {1.0: {}, 2.0: {}}
    
    for ts in TRAINING_SIZES:
        # Load ratio 1.0
        file_1 = AUG_RATIO_1_DIR / f"results_polypmix_ratio_1.0_ts_{ts}.txt"
        if file_1.exists():
            data[1.0][ts] = parse_result_file(file_1)
        
        # Load ratio 2.0
        file_2 = AUG_RATIO_2_DIR / f"results_polypmix_ratio_2.0_ts_{ts}.txt"
        if file_2.exists():
            data[2.0][ts] = parse_result_file(file_2)
    
    return data


def plot_metric_comparison_by_size(data, metric_key, metric_name, ylabel, filename):
    """Plot a metric comparison across training sizes for both ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, ts in enumerate(TRAINING_SIZES):
        ax = axes[idx]
        
        for ratio in [1.0, 2.0]:
            if ts in data[ratio]:
                d = data[ratio][ts]
                ax.plot(d['epochs'], d[metric_key], 
                       color=COLORS[ratio], linewidth=2, 
                       label=f'Ratio {ratio}', alpha=0.8)
                
                # Mark best epoch for validation metrics
                if 'val' in metric_key and d['best_epoch']:
                    best_idx = d['best_epoch'] - 1
                    if best_idx < len(d[metric_key]):
                        ax.axvline(x=d['best_epoch'], color=COLORS[ratio], 
                                  linestyle='--', alpha=0.5)
        
        ax.set_title(f'Training Size: {ts}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the 6th subplot
    axes[5].set_visible(False)
    
    plt.suptitle(f'{metric_name} Comparison: Ratio 1.0 vs Ratio 2.0', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_best_dice_comparison(data):
    """Plot bar chart comparing best dice scores across training sizes."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(TRAINING_SIZES))
    width = 0.35
    
    # Get best dice scores
    dice_1 = [data[1.0][ts]['best_dice'] if ts in data[1.0] else 0 for ts in TRAINING_SIZES]
    dice_2 = [data[2.0][ts]['best_dice'] if ts in data[2.0] else 0 for ts in TRAINING_SIZES]
    
    bars1 = ax.bar(x - width/2, dice_1, width, label='Ratio 1.0', color=COLORS[1.0], alpha=0.8)
    bars2 = ax.bar(x + width/2, dice_2, width, label='Ratio 2.0', color=COLORS[2.0], alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars1, dice_1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
               f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, dice_2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
               f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Best Validation Dice Score', fontsize=12)
    ax.set_title('Best Validation Dice Score Comparison\nPolypMix Ratio 1.0 vs Ratio 2.0', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TRAINING_SIZES)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'best_dice_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: best_dice_comparison.png")


def plot_convergence_comparison(data):
    """Plot epochs to reach best dice score."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(TRAINING_SIZES))
    width = 0.35
    
    # Get best epochs
    epochs_1 = [data[1.0][ts]['best_epoch'] if ts in data[1.0] else 0 for ts in TRAINING_SIZES]
    epochs_2 = [data[2.0][ts]['best_epoch'] if ts in data[2.0] else 0 for ts in TRAINING_SIZES]
    
    bars1 = ax.bar(x - width/2, epochs_1, width, label='Ratio 1.0', color=COLORS[1.0], alpha=0.8)
    bars2 = ax.bar(x + width/2, epochs_2, width, label='Ratio 2.0', color=COLORS[2.0], alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, epochs_1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, epochs_2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Best Epoch', fontsize=12)
    ax.set_title('Convergence Speed: Epoch to Reach Best Dice Score\nPolypMix Ratio 1.0 vs Ratio 2.0', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TRAINING_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: convergence_comparison.png")


def plot_training_curves_per_size(data):
    """Create detailed training curves for each training size."""
    for ts in TRAINING_SIZES:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = [
            ('train_loss', 'Training Loss', 'Loss'),
            ('val_loss', 'Validation Loss', 'Loss'),
            ('train_dice', 'Training Dice', 'Dice Score'),
            ('val_dice', 'Validation Dice', 'Dice Score'),
            ('train_iou', 'Training IoU', 'IoU Score'),
            ('val_iou', 'Validation IoU', 'IoU Score'),
        ]
        
        for ax, (key, title, ylabel) in zip(axes.flatten(), metrics):
            for ratio in [1.0, 2.0]:
                if ts in data[ratio]:
                    d = data[ratio][ts]
                    ax.plot(d['epochs'], d[key], color=COLORS[ratio], 
                           linewidth=2, label=f'Ratio {ratio}', alpha=0.8)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Curves - Training Size: {ts}\nPolypMix Ratio Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'training_curves_ts_{ts}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: training_curves_ts_{ts}.png")


def plot_ratio_comparison_within_approach(data):
    """Plot how each ratio performs across different training sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, ratio in enumerate([1.0, 2.0]):
        ax = axes[idx]
        
        for ts in TRAINING_SIZES:
            if ts in data[ratio]:
                d = data[ratio][ts]
                ax.plot(d['epochs'], d['val_dice'], linewidth=2, 
                       label=f'TS={ts}', alpha=0.8)
        
        ax.set_title(f'Augmentation Ratio {ratio}\nValidation Dice Across Training Sizes', 
                    fontweight='bold', fontsize=13)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Dice Score')
        ax.legend(title='Training Size')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.65, 0.95)
    
    plt.suptitle('Training Size Effect on Validation Dice for Each Augmentation Ratio', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'size_effect_per_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: size_effect_per_ratio.png")


def plot_summary_heatmap(data):
    """Create a heatmap summary of best dice scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create data matrix
    matrix = []
    for ratio in [1.0, 2.0]:
        row = []
        for ts in TRAINING_SIZES:
            if ts in data[ratio]:
                row.append(data[ratio][ts]['best_dice'])
            else:
                row.append(0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.85, vmax=0.95)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best Validation Dice Score', fontsize=11)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(TRAINING_SIZES)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(TRAINING_SIZES)
    ax.set_yticklabels(['Ratio 1.0', 'Ratio 2.0'])
    ax.set_xlabel('Training Size', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(len(TRAINING_SIZES)):
            text = ax.text(j, i, f'{matrix[i, j]:.4f}',
                          ha='center', va='center', color='black', fontsize=11, fontweight='bold')
    
    ax.set_title('Best Validation Dice Score Heatmap\nPolypMix Algorithm Comparison', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dice_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: dice_heatmap.png")


def create_summary_report(data):
    """Create a text summary report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PolyMix Algorithm Comparison Report")
    report_lines.append("Augmentation Ratio 1.0 vs Ratio 2.0")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    report_lines.append("BEST VALIDATION DICE SCORES:")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Training Size':<15}{'Ratio 1.0':<15}{'Ratio 2.0':<15}{'Winner':<15}")
    report_lines.append("-" * 50)
    
    for ts in TRAINING_SIZES:
        dice_1 = data[1.0][ts]['best_dice'] if ts in data[1.0] else 0
        dice_2 = data[2.0][ts]['best_dice'] if ts in data[2.0] else 0
        
        if dice_1 > dice_2:
            winner = "Ratio 1.0"
        elif dice_2 > dice_1:
            winner = "Ratio 2.0"
        else:
            winner = "Tie"
        
        report_lines.append(f"{ts:<15}{dice_1:.4f}         {dice_2:.4f}         {winner}")
    
    report_lines.append("")
    report_lines.append("CONVERGENCE (Best Epoch):")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Training Size':<15}{'Ratio 1.0':<15}{'Ratio 2.0':<15}")
    report_lines.append("-" * 50)
    
    for ts in TRAINING_SIZES:
        epoch_1 = data[1.0][ts]['best_epoch'] if ts in data[1.0] else 0
        epoch_2 = data[2.0][ts]['best_epoch'] if ts in data[2.0] else 0
        report_lines.append(f"{ts:<15}{epoch_1:<15}{epoch_2:<15}")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Save report
    report_path = OUTPUT_DIR / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved: comparison_report.txt")
    print('\n'.join(report_lines))


def main():
    print("Loading experiment data...")
    data = load_all_data()
    
    print("\nGenerating comparison plots...")
    
    # 1. Validation Dice comparison across training sizes
    plot_metric_comparison_by_size(data, 'val_dice', 'Validation Dice', 
                                   'Dice Score', 'val_dice_by_size.png')
    
    # 2. Validation Loss comparison
    plot_metric_comparison_by_size(data, 'val_loss', 'Validation Loss', 
                                   'Loss', 'val_loss_by_size.png')
    
    # 3. Training Dice comparison
    plot_metric_comparison_by_size(data, 'train_dice', 'Training Dice', 
                                   'Dice Score', 'train_dice_by_size.png')
    
    # 4. Best dice score bar chart
    plot_best_dice_comparison(data)
    
    # 5. Convergence comparison
    plot_convergence_comparison(data)
    
    # 6. Detailed training curves per training size
    plot_training_curves_per_size(data)
    
    # 7. Training size effect within each ratio
    plot_ratio_comparison_within_approach(data)
    
    # 8. Summary heatmap
    plot_summary_heatmap(data)
    
    # 9. Text summary report
    print("\n")
    create_summary_report(data)
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
