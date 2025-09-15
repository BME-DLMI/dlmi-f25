import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation


def optimize(func, optimizer):
    """
    Peforms optimization of the func function using the provided Pytorch optimizer

    Inputs:
    - func (function): function to be optimized
    - optimzier (torch.optim.Optimizer): Pytorch optimizer instance

    Returns:
    - xsnp (1D NumPy array): history of weight values
    - ysnp (1D NumPy array): history of function values

    """
    param = optimizer.param_groups[0]["params"][0]
    # initialize history with initial value
    y = func(param)
    xs = [param.detach().clone()]
    ys = [y.detach().clone()]

    # optimization loop
    for i in range(60):
        optimizer.zero_grad()
        loss = func(param)  # calculate loss
        loss.backward()  # compute gradients
        optimizer.step()  # update w

        # keep history for plotting
        xs.append(param.detach().clone())
        ys.append(func(param).detach().clone())

    # Convert pytorch tensors to numpy
    xsnp = np.stack([x.numpy() for x in xs]).squeeze()
    ysnp = np.stack([y.numpy() for y in ys]).squeeze()

    return xsnp, ysnp


def save_gif(func, x, y, fn):
    """
    Creats GIF of optimization trajectory

    Inputs:
    - func (function): function to be optimized
    - x (1D NumPy array): values of weight during training
    - y (1D NumPy array): corresponding function value y = func(x)
    - fn (string): filename to save GIF

    Returns:
    - anim (annimation.FuncAnimation): annimation.FuncAnimation object
    """

    xvec = torch.linspace(0, 10, 100)
    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(4, 3))
    ax.plot(xvec, func(xvec), "k", zorder=1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$L(w)$")
    scat = ax.scatter([], [], c="r", zorder=2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim([0, 300])
    ax.set_xticks([])
    ax.set_yticks([])

    def animate(i):
        scat.set_offsets(np.c_[x[i], y[i]])
        ax.set_title(f"step {i}")
        return (scat,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(x), interval=750, blit=False, repeat=True
    )
    anim.save(fn, fps=5)
    plt.show()
    return anim
