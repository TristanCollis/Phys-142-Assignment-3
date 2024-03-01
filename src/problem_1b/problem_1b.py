from typing import Any
import numpy as np
import constants as const


def run(
    Es: np.ndarray[float, Any], psis: np.ndarray[complex, Any]
) -> np.ndarray[float, Any]:

    E_0, E_1 = Es

    psi_0, psi_1 = psis

    psi_t = 2**-0.5 * (
        np.exp(-1j * E_0 * const.T).reshape(1, 1, -1) * psi_0
        + np.exp(-1j * E_1 * const.T).reshape(1, 1, -1) * psi_1
    )

    psi_pdf: np.ndarray[float, Any] = np.abs(psi_t) ** 2

    return psi_pdf
