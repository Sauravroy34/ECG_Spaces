"""
1D Grad-CAM Utilities
=====================

Computes Grad-CAM heatmaps for 1D signals and returns matplotlib
figures suitable for inline display in Streamlit.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def get_feature_importance(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int,
) -> np.ndarray:
    """
    Compute a 1D Grad-CAM saliency map.

    Parameters
    ----------
    model : nn.Module
        The full classification model.
    target_layer : nn.Module
        The convolutional layer to hook into (e.g. model.model.layer4[2].conv3).
    inputs : torch.Tensor
        Input tensor of shape (1, 1, 500).
    target_index : int
        Class index to compute gradients for.

    Returns
    -------
    np.ndarray
        Normalised importance array with the same length as the input signal.
    """
    model.eval()
    model.zero_grad()

    cache: dict = {}

    def forward_hook(module, input, output):
        output.retain_grad()
        cache["activations"] = output

    hook = target_layer.register_forward_hook(forward_hook)

    out = model(inputs)
    out[0, target_index].backward()

    activations = cache["activations"]
    grads = activations.grad

    weights = torch.mean(grads, dim=2, keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = torch.relu(cam)

    cam = cam.unsqueeze(1)
    size = inputs.shape[-1]
    cam_resized = F.interpolate(cam, size=size, mode="linear")

    cam_resized = cam_resized.squeeze().detach().cpu().numpy()

    cam_normalized = (cam_resized - np.min(cam_resized)) / (
        np.max(cam_resized) - np.min(cam_resized) + 1e-8
    )

    hook.remove()

    return cam_normalized


def create_gradcam_figure(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int,
    class_name: str = "",
) -> plt.Figure:
    """
    Create a publication-quality Grad-CAM overlay figure.

    Returns a matplotlib Figure (does NOT call plt.show()).
    """
    cam_normalized = get_feature_importance(model, target_layer, inputs, target_index)

    original_signal = inputs[0, 0].detach().cpu().numpy()
    seq_length = inputs.shape[-1]

    # ── Dark-themed figure ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    # Plot ECG signal
    ax.plot(original_signal, color="#00D4FF", linewidth=1.2, zorder=3, alpha=0.9)
    ax.set_xlim(0, seq_length)
    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.08
    ymin -= margin
    ymax += margin
    ax.set_ylim(ymin, ymax)

    # Heatmap overlay
    # Custom colormap: transparent-blue → cyan → yellow → red
    colors_list = [
        (0.0, 0.0, 0.15, 0.0),   # transparent dark
        (0.0, 0.3, 0.6, 0.25),   # dark blue
        (0.0, 0.8, 0.8, 0.45),   # cyan
        (1.0, 0.85, 0.0, 0.6),   # yellow
        (1.0, 0.2, 0.1, 0.75),   # red
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("gradcam", colors_list, N=256)

    heatmap_data = cam_normalized.reshape(1, -1)
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap=cmap,
        extent=[0, seq_length, ymin, ymax],
        alpha=0.7,
        zorder=2,
    )

    # Title & labels
    title = "1D Grad-CAM — Feature Importance"
    if class_name:
        title += f"  ·  Target: {class_name}"
    ax.set_title(title, color="#E0E0E0", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sample Index", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Amplitude", color="#AAAAAA", fontsize=10)

    # Style spines & ticks
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(colors="#888888", labelsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Importance", color="#AAAAAA", fontsize=10)
    cbar.ax.tick_params(colors="#888888", labelsize=8)
    cbar.outline.set_edgecolor("#333333")

    plt.tight_layout()
    return fig
