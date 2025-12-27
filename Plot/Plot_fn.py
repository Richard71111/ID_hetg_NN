import matplotlib.pyplot as plt
import numpy as np
import math

def plot_trajectory(
    x, y, idx,
    title='Voltage',
    xlabel='Time (ms)',
    ylabel='Voltage (mV)',
    max_cols=3
):
    """
    x: (T,)
    y: (T, N)
    idx: list or array of indices to plot
    max_cols: maximum number of columns in subplot layout
    """
    idx = np.asarray(idx)
    n = len(idx)

    if n == 0:
        raise ValueError("idx must contain at least one index.")

    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False
    )

    axs = axs.flatten()

    for ax, i in zip(axs, idx):
        ax.plot(x, y[:, i], label=f'Cell {i}')
        ax.set_title(f'{title} - Cell {i}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    for ax in axs[len(idx):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def plot_2D_map(
    time,
    data,
    xlabel="Cell Number",
    ylabel="Time (ms)",
    zlabel="Voltage (mV)",
    title=None,
    cmap="viridis",
    figsize=(10, 6)
):
    """
    time: (T,) array-like
    data: (T, N) array-like
    """

    time = np.asarray(time)
    data = np.asarray(data)

    if data.shape[0] != len(time):
        raise ValueError(
            f"time length ({len(time)}) must match data.shape[0] ({data.shape[0]})"
        )

    T, N = data.shape

    # meshgrid: X -> cell index, Y -> time
    X, Y = np.meshgrid(
        np.arange(N),
        time
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, data,
        cmap=cmap,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if title is not None:
        ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=12)

    plt.tight_layout()
    plt.show()
