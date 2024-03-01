from typing import Any
from matplotlib import pyplot as plt
import constants as const
import helpers
import numpy as np


def power_iteration(
    A: np.ndarray[complex, Any], sigma: float = 0, iterations: int = 100
) -> tuple[float, np.ndarray[complex, Any]]:
    A_shift_inverse = np.linalg.inv(A - sigma * np.eye(A.shape[-1]))
    u = np.random.random(A.shape[-1]).reshape(1, -1, 1)

    for _ in range(iterations):
        u = helpers.normalize(A_shift_inverse @ u)

    # lambda_u = float(np.sum(np.conj(u) * (A_shift_inverse @ u)) / np.sum(np.conj(u) * u))
    # lambda_u = np.mean((A @ u) / u).astype(float)

    lambda_u = (
        float(np.sum(np.conj(u) * u) / (np.sum(np.conj(u) * (A_shift_inverse @ u))))
        + sigma
    )

    return lambda_u, u


def run(save: bool = False, show: bool = False, print_output: bool = False) -> tuple:

    H = helpers.H_matrix()

    # Es, psis = scipy.sparse.linalg.eigsh(np.squeeze(H), k=2, sigma=0, which="LM")
    # psi_0 = np.squeeze(helpers.normalize(psis[:, 0].reshape(1, -1, 1)))
    # psi_1 = np.squeeze(helpers.normalize(psis[:, 1].reshape(1, -1, 1)))

    E_0, psi_0 = power_iteration(H, sigma=1.2, iterations=100)
    E_1, psi_1 = power_iteration(H, sigma=1.4, iterations=100)

    if save or show:

        plt.figure(figsize=(8, 4.5))
        plt.xlim([const.X_0, const.X_ND])
        plt.ylim([0, 3])

        plt.plot(const.X.flat, helpers.V(const.X).flat, "--")
        plt.plot(const.X.flat, np.squeeze(psi_0) + E_0, "b-")
        # plt.plot(const.X.flat, np.abs(const.PSI_SYMMETRIC.flat)**2 + E_0, 'b--')
        plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_0, "b--")
        plt.plot(const.X.flat, np.squeeze(psi_1) + E_1, "r-")
        # plt.plot(const.X.flat, np.abs(const.PSI_ASYMMETRIC.flat)**2 + E_1, 'r--')
        plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_1, "r--")

    if save:
        plt.savefig(fname="problem_1a.png")

    if show:
        plt.show()

    if print_output:
        print(f"E_0: {E_0}")
        print(f"E_1: {E_1}")
        print(f"delta_E: {E_1 - E_0}")

    return (E_0, E_1), (psi_0, psi_1)
