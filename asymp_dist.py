"""Module for evaluating chaos in dimension-preserving input-output functions"""

import numpy as np

def asymp_dist(func, input, epsilon, n_iter=500):
    """Estimate Lyapunov exponent of a dimension-preserving input-output 
    function by computing the distance between trajectories with small initial 
    distance epsilon. Initial distance is created using random permutation of n 
    values in input, where n = `epsilon` * len(`input`).
    
    Parameters
    ----------
    func : function
        A function where the output vector is of the same size as the input
        vector.
    input : 1d array
        An input vector, e.g. an image.
    epsilon : float
        Fraction of values to permute in input to create the small initial 
        distance between trajectories. Between 0 and 1.
    n_iter : int
        Number of iterations of `func` before taking the distance estimate.

    Returns
    -------
    estimate : float
        An estimate of the Lyapunov exponent of the function `func`.

    Reference
    ---------
    Feng, Zhang & Lai (2019) arXiv
    """
    x_n = y_n = input

    arr = np.arange(len(input))
    n_permute = max(2, int(len(input) * epsilon))
    idx_permute = np.random.choice(arr, size=n_permute, replace=False)
    y_n[idx_permute] = np.random.permutation(y_n[idx_permute])

    for i in n_iter:
        x_n = func(x_n)
        y_n = func(y_n)

    return np.abs(y_n - x_n)

    

    