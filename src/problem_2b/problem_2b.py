from typing import Any
from matplotlib import pyplot as plt
import numpy as np

from helpers.helpers import mcmc_full


def run(
    N: int,
    temperatures: np.ndarray[float, Any],
    steps: int,
    burn_in: int,
    show: bool = False,
    save: bool = False,
) -> None:

    m_means = np.empty(temperatures.shape[0])
    m_stds = np.empty(temperatures.shape[0])

    for i, temp in enumerate(temperatures):
        lattice = np.ones((N, N))

        m_values = mcmc_full(lattice, temp, steps)[:burn_in]

        m_means[i] = np.mean(m_values)
        m_stds[i] = np.std(m_values)

    critical_temperature = 2 / np.log(1 + np.sqrt(2))
    plt.figure()
    plt.errorbar(temperatures, m_means, yerr=m_stds)
    plt.plot(
        [critical_temperature, critical_temperature],
        [0, 1],
        "--",
        color="black",
        label=f"Tc={critical_temperature:.2f}",
    )
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")

    if show:
        plt.show()

    if save:
        plt.savefig(fname="problem_2b.png")

    plt.clf()
