import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import glob
import imageio.v2 as imageio  # safer for compatibility


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_ray_frame(positions,
                        bearings,
                        x_hat,
                        x_true=None,
                        sigmas=None,
                        step=None,
                        save_path=None,
                        ray_length=50.0):
    """
    positions : (K, 2)  sensor positions
    bearings  : (K,)      for single source
                (K, M)    for multiple sources  (K=sensors, M=sources)
                [radians]
    x_hat     : (2,)      for single source
                (M, 2)    for multiple sources
    x_true    : same shape as x_hat or None
    sigmas    : same shape as bearings (K,) or (K, M) or None
                std-dev of angle error [radians]
    ray_length: radius of rays / sectors
    """

    # ----- to numpy -----
    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    positions = to_np(positions)
    bearings  = to_np(bearings)
    x_hat     = to_np(x_hat)
    x_true    = to_np(x_true)
    sigmas    = to_np(sigmas)

    # bearings: (K,) -> (K,1)
    if bearings.ndim == 1:
        bearings = bearings[:, None]      # (K, 1)
    elif bearings.ndim != 2:
        raise ValueError(f"bearings must be (K,) or (K,M), got {bearings.shape}")

    K, M = bearings.shape  # K sensors, M sources

    # x_hat: (2,) -> (1,2)
    if x_hat.ndim == 1:
        x_hat = x_hat[None, :]            # (1,2)
    elif x_hat.ndim != 2:
        raise ValueError(f"x_hat must be (2,) or (M,2), got {x_hat.shape}")
    if x_hat.shape[0] != M:
        raise ValueError(f"Mismatch: bearings has M={M} sources, x_hat has {x_hat.shape[0]}")

    # x_true (optional)
    if x_true is not None:
        if x_true.ndim == 1:
            x_true = x_true[None, :]
        elif x_true.ndim != 2:
            raise ValueError(f"x_true must be (2,) or (M,2), got {x_true.shape}")
        if x_true.shape[0] != M:
            raise ValueError(f"Mismatch: bearings has M={M} sources, x_true has {x_true.shape[0]}")

    # sigmas: same broadcasting as bearings
    if sigmas is not None:
        if sigmas.ndim == 1:
            sigmas = sigmas[:, None]          # (K,1)
        elif sigmas.ndim != 2:
            raise ValueError(f"sigmas must be (K,) or (K,M), got {sigmas.shape}")
        if sigmas.shape != bearings.shape:
            raise ValueError(f"sigmas shape {sigmas.shape} must match bearings {bearings.shape}")

    # positions: (K,2)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must be (K,2), got {positions.shape}")
    if positions.shape[0] != K:
        raise ValueError(f"positions has K={positions.shape[0]} sensors, bearings has K={K}")

    # ----- plotting -----
    fig, ax = plt.subplots(figsize=(14, 6))

    # sensors
    ax.plot(positions[:, 0], positions[:, 1], 'ko', label='Sensors')

    # colors per source
    colors = plt.cm.tab10(np.linspace(0, 1, max(M, 1)))

    est_label_used = [False] * M
    true_label_used = [False] * M
    sector_label_used = [False] * M

    # loop over sensors, then sources
    for k in range(K):
        x_s, y_s = positions[k]
        x_s = float(x_s)
        y_s = float(y_s)

        for m in range(M):
            theta = float(bearings[k, m])   + (np.pi / 2)# radians +
            color = colors[m]

            # central ray
            dx, dy = np.cos(theta), np.sin(theta)
            ax.plot(
                [x_s, x_s + ray_length * dx],
                [y_s, y_s + ray_length * dy],
                linestyle='--',
                color=color,
                alpha=0.5
            )

            # uncertainty sector (if sigmas provided)
            if sigmas is not None:
                sigma = float(sigmas[k, m])   # radians
                theta_deg = np.degrees(theta)
                sigma_deg = np.degrees(sigma)

                wedge = patches.Wedge(
                    center=(x_s, y_s),
                    r=ray_length,
                    theta1=theta_deg - sigma_deg,
                    theta2=theta_deg + sigma_deg,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.15
                )
                ax.add_patch(wedge)
                if not sector_label_used[m]:
                    wedge.set_label(f'σ sector source {m}')
                    sector_label_used[m] = True

    # plot sources
    for m in range(M):
        color = colors[m]
        # estimated
        ax.plot(
            x_hat[m, 0], x_hat[m, 1],
            marker='o',
            color=color,
            markersize=8,
            label=f'Estimated source {m}' if not est_label_used[m] else None
        )
        est_label_used[m] = True

        # true
        if x_true is not None:
            ax.plot(
                x_true[m, 0], x_true[m, 1],
                marker='*',
                color=color,
                markersize=12,
                linestyle='None',
                label=f'True source {m}' if not true_label_used[m] else None
            )
            true_label_used[m] = True

    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    if step is not None:
        ax.set_title(f"Step {step}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
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
