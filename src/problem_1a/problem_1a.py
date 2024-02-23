from matplotlib import pyplot as plt
import constants as const
import helpers
import scipy.sparse.linalg
import numpy as np

from helpers.helpers import Laplace_matrix


def run(save: bool = False, show: bool = False):
    H = helpers.H_matrix()

    Es, psis = scipy.sparse.linalg.eigsh(np.squeeze(H), k=2, sigma=0, which="LM")

    E_0 = Es[0]
    E_1 = Es[1]

    psi_0 = np.squeeze(helpers.normalize(psis[:, 0].reshape(1, -1, 1)))
    psi_1 = np.squeeze(helpers.normalize(psis[:, 1].reshape(1, -1, 1)))

    if not (save or show):
        return

    plt.figure(figsize=(8, 4.5))
    plt.xlim([const.X_0, const.X_ND])
    plt.ylim([0, 3])

    plt.plot(const.X.flat, helpers.V(const.X).flat, '--')
    plt.scatter(const.X.flat, psi_0 + E_0)
    # plt.plot(const.X.flat, np.abs(const.PSI_SYMMETRIC.flat)**2 + E_0, 'b--')
    plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_0, 'b--')
    plt.scatter(const.X.flat, psi_1 + E_1)
    # plt.plot(const.X.flat, np.abs(const.PSI_ASYMMETRIC.flat)**2 + E_1, 'r--')
    plt.plot(const.X.flat, np.ones_like(const.X.flat) * E_1, 'r--')

    print(E_0)
    print(E_1)

    if save:
        plt.savefig(fname="problem_1a.png")

    if show:
        plt.show()
