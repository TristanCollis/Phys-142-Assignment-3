from typing import Any
import numpy as np
import constants as const
from helpers import animate, K_0, normalize


def run(
    Es: np.ndarray[float, Any],
    psis: np.ndarray[complex, Any],
) -> None:

    E_0, E_1 = Es

    psi_0, psi_1 = psis

    psi_schrodinger = 2**-0.5 * (
        np.exp(-1j * E_0 * const.T).reshape(1, 1, -1) * psi_0
        + np.exp(-1j * E_1 * const.T).reshape(1, 1, -1) * psi_1
    )

    pdf_schrodinger = np.abs(psi_schrodinger) ** 2

    animate(
        np.squeeze(pdf_schrodinger).T,
        np.squeeze(const.X),
        "problem_1c_schrodinger.mp4",
    )

    psi_initial = (psi_0 + psi_1) / np.sqrt(2)

    psi_propagator = np.zeros((const.NT, *psi_initial.shape), dtype=complex)
    psi_propagator[0] = normalize(psi_0 + psi_1)

    for i in range(1, const.NT):
        psi_propagator[i] = normalize(K_0 @ psi_propagator[i - 1])

    pdf_propagator = np.abs(psi_propagator) ** 2

    animate(
        np.squeeze(pdf_propagator), np.squeeze(const.X), "problem_1c_propagator.mp4"
    )
