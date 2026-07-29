"""Matplotlib visualization helpers."""

import matplotlib.pyplot as plt


def show_stages(images, titles, save_path=None, show=True):
    """Display up to N images side by side in a 1xN subplot.

    Args:
        images: List of images (RGB arrays, or grayscale arrays).
        titles: List of subplot titles, one per image.
        save_path: If provided, save the figure to this path.
        show: If True, call ``plt.show()``.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        # 2D arrays are grayscale; render them with a gray colormap.
        cmap = "gray" if image.ndim == 2 else None
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()

    return fig
