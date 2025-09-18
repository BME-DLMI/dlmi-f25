import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from matplotlib import cm
import PIL.Image


def callback_plot(index, X, Y, model, device, output_dir):

    plot_title = f"Iteration: {index:05}"
    file_name = f"model_{index:05}.png"
    file_path = os.path.join(output_dir, file_name)

    GRID_X_START = -1.5
    GRID_X_END = 2.5
    GRID_Y_START = -1.0
    GRID_Y_END = 2

    grid = np.mgrid[GRID_X_START:GRID_X_END:100j, GRID_Y_START:GRID_Y_END:100j]
    grid_2d = grid.reshape(2, -1).T
    XX, YY = grid
    all_inputs = torch.from_numpy(grid_2d.astype(np.float32)).to(device)

    logit = model(all_inputs)
    e_x = torch.exp(logit - torch.max(logit, dim=1, keepdim=True)[0])
    prob = e_x / torch.sum(e_x, dim=1, keepdim=True)
    prob = prob.detach().cpu()
    prob = prob[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(plot_title, fontsize=30)
    ax.axis("off")
    if XX is not None and YY is not None and prob is not None:
        ax.contourf(
            XX, YY, prob.reshape(XX.shape), 25, alpha=1, cmap=cm.coolwarm
        )  # cmap=cm.Spectral)
        ax.contour(
            XX, YY, prob.reshape(XX.shape), levels=[0.5], cmap="Greys", vmin=0, vmax=0.6
        )
    ax.scatter(
        X[:, 0], X[:, 1], c=Y.ravel(), s=40, cmap=plt.cm.coolwarm, edgecolors="black"
    )
    if file_path:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()


def create_gif(image_paths, output_gif_path, duration=500):
    """Creates a GIF from a list of PNG image paths."""

    images = [PIL.Image.open(image_path) for image_path in image_paths]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0,  # 0 means infinite loop
    )
