# cram/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional

def plot_retention_heatmap(retention_weights: np.ndarray, title: Optional[str] = None):
    """Plot retention mechanism attention weights"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        retention_weights,
        cmap='viridis',
        xticklabels=5,
        yticklabels=5
    )
    if title:
        plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    return plt.gcf()

def plot_training_metrics(
    steps: List[int],
    losses: List[float],
    perplexities: List[float],
    save_path: Optional[str] = None
):
    """Plot training metrics over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(steps, losses, label='Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Time')
    ax1.grid(True)
    
    # Plot perplexity
    ax2.plot(steps, perplexities, label='Perplexity', color='orange')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    return fig