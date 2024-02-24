from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import constants as const
from helpers.helpers import expectation, x_operator

def run(Es: np.ndarray[float, Any], psis: np.ndarray[complex, Any], show: bool = False, print_output: bool = False, save: bool = False) -> None:
    if not (save or show or print_output):
        return None
    
    E_0, E_1 = Es

    psi_0 = np.expand_dims(psis[:, 0], 0)
    psi_1 = np.expand_dims(psis[:, 1], 0)

    t = np.expand_dims(const.T, 1)

    Psi = np.expand_dims(
        0.5 ** 2 * (
        np.exp(-1j * E_0 * t) * psi_0
        + np.exp(-1j * E_1 * t) * psi_1
        ),
        0
    )

    x_bar = expectation(x_operator, Psi.transpose((0, 2, 1)))

    plt.plot(const.T.flat, x_bar.flat)

    if show:
        plt.show()

    if save:
        plt.savefig(fname=f"{__name__.split(".")[-1]}.png")