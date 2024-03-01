from matplotlib import pyplot as plt
import numpy as np
from helpers import mcmc_full


def run(N: int, temperature: float, steps: int, show: bool = False, save: bool = False) -> None:

    lattice_up = np.ones((N, N))

    magnetization_up = mcmc_full(lattice_up, temperature, steps)

    plt.plot(magnetization_up)

    if show:
        plt.show()

    if save:
        plt.savefig(fname="problem_2a_up.png")

    plt.clf()

    lattice_random = np.random.randint(0, 3, size=(N,N)) - 1

    magnetization_random = mcmc_full(lattice_random, temperature, steps)

    plt.plot(magnetization_random)

    if show:
        plt.show()

    if save:
        plt.savefig(fname="problem_2a_random.png")

    plt.clf()