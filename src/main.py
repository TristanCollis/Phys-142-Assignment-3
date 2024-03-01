import numpy as np
import problem_1a
import problem_1b
import problem_1c

import problem_2a
import problem_2b

# Es, psis = problem_1a.run(save=True)
# psi_pdf = problem_1b.run(Es, psis)
# problem_1c.run(Es, psis)

# problem_2a.run(N=20, temperature=2, steps=200_000)
problem_2b.run(N=20, temperatures=np.arange(1, 4, 0.2), burn_in=100_000, steps=200_000, save=True)