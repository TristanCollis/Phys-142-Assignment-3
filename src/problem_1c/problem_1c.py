import numpy as np
import scipy.sparse.linalg

import constants as const
from helpers import animate, K_0, normalize, H_matrix


def run(filename: str = "problem_1c.mp4") -> None:
    H = H_matrix()

    Es, psis = scipy.sparse.linalg.eigsh(np.squeeze(H), k=2, sigma=0, which="LM")

    E_0 = Es[0]
    E_1 = Es[1]

    psi_0 = normalize(psis[:, 0].reshape(1, -1, 1)).reshape(1, -1)
    psi_1 = normalize(psis[:, 1].reshape(1, -1, 1)).reshape(1, -1)

    psi_t = 2**-0.5 * (
        np.exp(-1j * E_0 * const.T).reshape(-1, 1) * psi_0
        + np.exp(-1j * E_1 * const.T).reshape(-1, 1) * psi_1
    )

    psi_pdf = np.abs(psi_t) ** 2

    animate(np.squeeze(psi_pdf), np.squeeze(const.X), filename)
