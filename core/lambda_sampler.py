import numpy as np


def sample_lambda(num_obj: int, rng=np.random, restrict_to_extremes: bool = False) -> np.ndarray:
    """
    Corner-biased λ sampler for multi-objective preference vectors on the simplex.

    Mixture:
      30% pure corners (one-hot)
      20% 2-active edges (two non-zero components summing to 1)
      20% sparse Dirichlet(0.3)  — still near-edge
      30% uniform Dirichlet(1.0) — interior

    When restrict_to_extremes=True, the interior Dirichlet is replaced by more edge
    samples, producing only corners + edges + sparse Dirichlet (used for the A.4
    in-distribution generalization test: train without interior, eval on interior).
    """
    r = rng.uniform()
    if r < 0.30:
        v = np.zeros(num_obj)
        v[rng.randint(num_obj)] = 1.0
    elif r < 0.50:
        v = np.zeros(num_obj)
        i, j = rng.choice(num_obj, size=2, replace=False)
        t = rng.uniform()
        v[i], v[j] = t, 1.0 - t
    elif r < 0.70:
        v = rng.dirichlet(0.3 * np.ones(num_obj))
    else:
        if restrict_to_extremes:
            v = np.zeros(num_obj)
            i, j = rng.choice(num_obj, size=2, replace=False)
            t = rng.uniform()
            v[i], v[j] = t, 1.0 - t
        else:
            v = rng.dirichlet(1.0 * np.ones(num_obj))
    return v
