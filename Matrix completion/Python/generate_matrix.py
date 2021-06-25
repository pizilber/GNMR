import numpy as np

"""
Code taken from R2RILS by Jonathan Bauch, Boaz Nadler and Pini Zilber
"""

def generate_matrix(n1, n2,  singular_values):
    """
    Generate a matrix with specific singular values
    :param int n1: number of rows
    :param int n2: number of columns
    :param list singular_values: required singular values
    """
    rank = len(singular_values)
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    U = np.linalg.qr(U)[0]
    V = np.linalg.qr(V)[0]
    return U @ np.diag(singular_values) @ V.T