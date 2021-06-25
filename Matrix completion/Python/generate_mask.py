import numpy as np

"""
Code taken from R2RILS by Jonathan Bauch, Boaz Nadler and Pini Zilber
"""

def generate_mask(n1, n2, rank, p):
    """
    Generate a mask with at least r observed entries in each row and column (r == rank)
    In case p is too small, function might not return
    :param int n1: number of rows
    :param int n2: number of columns
    :param float p: probability of observing an entry
    :param int rank: rank of matrix
    """
    num_resamples = 0
    found = False
    while not found:
        num_resamples += 1
        omega = np.round(0.5 * (np.random.random((n1, n2)) + p))
        # make sure there are enough visible entries on rows and columns
        found = (min(np.count_nonzero(omega, axis=0)) >= rank) and min(np.count_nonzero(omega, axis=1)) >= rank
        if (num_resamples % 1e4 == 0):
              print('resampling mask {}'.format(num_resamples))
    return omega