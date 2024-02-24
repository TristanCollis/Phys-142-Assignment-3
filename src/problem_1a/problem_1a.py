from typing import Any
from matplotlib import pyplot as plt
import constants as const
import helpers
import scipy.sparse.linalg
import numpy as np

from helpers.helpers import Laplace_matrix


# def power_iteration(A: np.ndarray[complex, Any], iterations=10) -> np.ndarray[complex: Any]:
#     b_k = 



def run(save: bool = False, show: bool = False, print_output: bool = False) -> tuple:
    
    H = helpers.H_matrix()

    Es, psis = scipy.sparse.linalg.eigsh(np.squeeze(H), k=2, sigma=0, which="LM")

    E_0 = Es[0]
    E_1 = Es[1]

    psi_0 = np.squeeze(helpers.normalize(psis[:, 0].reshape(1, -1, 1)))
    psi_1 = np.squeeze(helpers.normalize(psis[:, 1].reshape(1, -1, 1)))


    if save or show:

        plt.figure(figsize=(8, 4.5))
        plt.xlim([const.X_0, const.X_ND])
        plt.ylim([0, 3])

        plt.plot(const.X.flat, helpers.V(const.X).flat, '--')
        plt.plot(const.X.flat, psi_0 + E_0, 'b-')
        # plt.plot(const.X.flat, np.abs(const.PSI_SYMMETRIC.flat)**2 + E_0, 'b--')
        plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_0, 'b--')
        plt.plot(const.X.flat, psi_1 + E_1, 'r-')
        # plt.plot(const.X.flat, np.abs(const.PSI_ASYMMETRIC.flat)**2 + E_1, 'r--')
        plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_1, 'r--')

    if save:
        plt.savefig(fname=f"{__name__.split(".")[-1]}.png")

    if show:
        plt.show()

    if print_output:
        print(f"E_0: {E_0}")
        print(f"E_1: {E_1}")
        print(f"delta_E: {E_1 - E_0}")
    
    return Es, psis