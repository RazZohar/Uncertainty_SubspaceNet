import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import glob
import imageio.v2 as imageio  # safer for compatibility

def visualize_ray_frame(positions, bearings, x_hat, x_true=None, step=None, save_path=None):
    positions = positions.detach().cpu().numpy()
    bearings  = bearings.detach().cpu().numpy()
#    x_hat     = x_hat.detach().cpu().numpy()
    if x_true is not None:
        x_true = x_true.detach().cpu().numpy()

    plt.figure(figsize=(14, 6))
    for (x, y), theta in zip(positions, bearings):
        x = float(x)
        y = float(y)
        theta = float(theta)
        dx, dy = np.cos(theta), np.sin(theta)
        plt.plot([x, x + 10 * dx], [y, y + 10 * dy], 'b--', alpha=0.5)
        plt.plot(x, y, 'ko')

    #plt.plot(x_hat[0], x_hat[1], 'ro', label='Estimated (x̂)', markersize=8)
    if x_true is not None:
        plt.plot(x_true[0], x_true[1], 'g*', label='Ground Truth', markersize=12)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    if step is not None:
        plt.title(f"Step {step}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def create_sample_gif(sample_idx: int, save_dir: str = "visualizations", gif_path: str = None, fps: int = 2):
    """
    Create a GIF from a sequence of ray intersection images.

    Args:
        sample_idx : int - Index of the sample tracked during training
        save_dir   : str - Directory containing PNGs
        gif_path   : str - Path to output GIF (if None, auto-generated)
        fps        : int - Frames per second
    """
    pattern = os.path.join(save_dir, f"sample_{sample_idx:03d}_epoch_*.png")
    frame_paths = sorted(glob.glob(pattern))

    if not frame_paths:
        raise FileNotFoundError(f"No images found for sample {sample_idx} in {save_dir}")

    if gif_path is None:
        gif_path = os.path.join(save_dir, f"sample_{sample_idx:03d}_evolution.gif")

    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"[GIF] Saved {gif_path}")
