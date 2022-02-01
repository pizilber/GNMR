### Gauss-Newton based algorithm for matrix sensing ###
### Written by Pini Zilber and Boaz Nadler, 2021 ###

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize

# initialization options
INIT_WITH_ATb = 0
INIT_WITH_RANDOM = 1
INIT_WITH_USER_DEFINED = 2

def GNMR_sensing(b, A, n1, n2, rank, verbose=True, alpha=1, max_outer_iter=100,
    max_inner_iter=2000, lsqr_init_tol=1e-15, lsqr_smart_tol=True, lsqr_smart_obj_min=1e-5,
    init_option=INIT_WITH_ATb, init_U=None, init_V=None,
    stop_relRes=1e-16, stop_relDiff = -1, stop_relResDiff = -1,
    r_projection_in_iteration=False):
    """
    Run GNMR algorithm for matrix completion
    :param ndarray b: observed linear measurements of the underlying matrix
    :param ndarray A: the sensing operator
    :paran int n1: number of rows in the underlying matrix
    :paran int n2: number of columns in the underlying matrix
    :param int rank: Underlying rank matrix
    :param bool verbose: if True, display intermediate results
    :param int alpha: indicating the variant of GNMR (e.g., 1: setting, 0: averaging, -1: updating)
    :param int max_outer_iter: Maximal number of outer iterations
    :param int max_inner_iter: Maximal number of inner iterations
    :param float lsqr_init_tol: initial tolerance of the LSQR solver
    :param bool lsqr_smart_tol: if True, when relRes <= lsqr_smart_obj_min, use lsqr_tol=objective**2
    :param float lsqr_smart_obj_min: relRes threshold to begin smart tolerance from
    :param int init_option: how to initialize U and V (INIT_WITH_SVD, INIT_WITH_RAND, or INIT_WITH_USER_DEFINED)
    :param ndarray init_U: U initialization (n1,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param ndarray init_V: V initialization (n2,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param float stop_relRes: relRes threshold for ealy stopping (relevant to noise-free case), -1 to disable
    :param float stop_relDiff: relative X_hat difference threshold for ealy stopping, -1 to disable
    :param float stop_relResDiff: relRes difference difference threshold for early stopping, -1 to disable
    :parma bool r_projection_in_iteration: if true, error estimation at each iteration
      is calculated for the best rank-r approximation using SVD
    :return: GNMR's estimate, final iteration number, convergence flag and all relRes
    """
    m = np.size(b)
    omega = np.ones((n1,n2))

    # initial estimate
    if init_option == INIT_WITH_ATb:
      X = A.T @ b
      X = X.reshape((n1,n2))
      (U, _, V) = linalg.svds(X, k=rank, tol=1e-16)
      V = V.T
    elif init_option == INIT_WITH_RANDOM:
      U = np.random.randn(n1, rank)
      V = np.random.randn(n2, rank)
      U = np.linalg.qr(U)[0]
      V = np.linalg.qr(V)[0]
    else:
      U = init_U
      V = init_V

    # generate sparse indices to accelerate future operations.
    sparse_matrix_rows, sparse_matrix_columns = generate_sparse_matrix_entries(A, rank, n1, n2)

    # before iterations
    early_stopping_flag = False
    relRes = 1
    all_relRes = [relRes]
    best_relRes = np.max(np.abs(X))
    X_hat = U @ V.T
    X_hat_best_2r = X_hat  # stores best intermediate rank 2r estimate

    # iterations
    iter_num = 0
    while iter_num < max_outer_iter and not early_stopping_flag:
        iter_num += 1
        
        # determine LSQR tolerance
        current_tol = lsqr_init_tol
        if lsqr_smart_tol and relRes < lsqr_smart_obj_min:
          current_tol = min(current_tol, relRes**2)
        
        # solve the least squares problem
        L = generate_sparse_L(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, n1, n2, rank)
        b_t = b + alpha * A @ np.ravel(U @ V.T)
        x, _, _, res = linalg.lsqr(A @ L, b_t, atol=current_tol, btol=current_tol, iter_lim=max_inner_iter)[:4]
        relRes = res / np.linalg.norm(b)
        x = convert_x_representation(x, rank, n1, n2)

        # obtain new estimates for U and V
        U_tilde, V_tilde = get_U_V_from_solution(x, rank, n1, n2)
        U_next = 0.5 * (1 - alpha) * U + U_tilde
        V_next = 0.5 * (1 - alpha) * V + V_tilde
        
        # get new estimate and calculate corresponding error
        X_hat_previous = X_hat
        X_hat_2r = U @ V_next.T + U_next @ V.T - U @ V.T
        if r_projection_in_iteration:
          (U_r, Sigma_r, V_r) = linalg.svds(X_hat_2r, k=rank, tol=1e-17)
          X_hat = U_r @ np.diag(Sigma_r) @ V_r
        else:
          X_hat = U_next @ V_next.T
        X_hat_diff =  np.linalg.norm(X_hat - X_hat_previous, ord='fro') / np.linalg.norm(X_hat, ord='fro')

        all_relRes.append(relRes)
        if relRes < best_relRes:
          best_relRes = relRes
          X_hat_best_2r = X_hat_2r

        # update U, V
        U = U_next
        V = V_next

        # report
        if verbose:
          print("[INSIDE GNMR] iter: " + str(iter_num) + ", relRes: " + str(relRes))

        # check early stopping criteria
        if stop_relRes > 0:
          early_stopping_flag |= relRes < stop_relRes
        if stop_relDiff > 0:
          early_stopping_flag |= X_hat_diff < stop_relDiff
        if stop_relResDiff > 0:
          early_stopping_flag |= np.abs(relRes / all_relRes[-2] - 1) < stop_relResDiff
        if verbose and early_stopping_flag:
          print("[INSIDE GNMR] early stopping")

    # return
    convergence_flag = iter_num < max_outer_iter
    (U_r, Sigma_r, V_r) = linalg.svds(X_hat_best_2r, k=rank, tol=1e-17)
    X_hat = U_r @ np.diag(Sigma_r) @ V_r
    return X_hat, iter_num, convergence_flag, all_relRes


# auxiliary functions
# (taken from the matrix completion algorithm, can probably be optimized for matrix sensing)

def convert_x_representation(x, rank, n1, n2):
    recovered_x = np.array([x[k * rank + i] for i in range(rank) for k in range(n2)])
    recovered_y = np.array([x[rank * n2 + j * rank + i] for i in range(rank) for j in range(n1)])
    return np.append(recovered_x, recovered_y)


def generate_sparse_matrix_entries(omega, rank, n1, n2):
    row_entries = []
    columns_entries = []
    row = 0
    for j in range(n1):
        for k in range(n2):
            if 0 != omega[j][k]:
                # add indices for U entries
                for l in range(rank):
                    columns_entries.append(k * rank + l)
                    row_entries.append(row)
                # add indices for V entries
                for l in range(rank):
                    columns_entries.append((n2 + j) * rank + l)
                    row_entries.append(row)
                row += 1
    return row_entries, columns_entries


def generate_sparse_L(U, V, omega, row_entries, columns_entries, n1, n2, rank):
    # we're generating row by row
    data_vector = np.concatenate(
        [np.concatenate([U[j], V[k]]) for j in range(n1) for k in range(n2) if 0 != omega[j][k]])
    return sparse.csr_matrix(sparse.coo_matrix((data_vector, (row_entries, columns_entries)),
                                               shape=(omega.size, rank * (n1 + n2))))


def get_U_V_from_solution(x, rank, n1, n2):
    VT = np.array([x[i * n2:(i + 1) * n2] for i in range(rank)])
    UT = np.array([x[rank * n2 + i * n1: rank * n2 + (i + 1) * n1] for i in range(rank)])
    return UT.T, VT.T


def get_estimated_value(x, U, V, rank, n1, n2):
    # calculate U's contribution
    estimate = np.sum(
        [np.dot(U.T[i].reshape(n1, 1), np.array(x[i * n2:(i + 1) * n2]).reshape(1, n2)) for i in
         range(rank)],
        axis=0)
    # calculate V's contribution
    estimate += np.sum(
        [np.dot(x[rank * n2 + i * n1: rank * n2 + (i + 1) * n1].reshape(n1, 1),
                V.T[i].reshape(1, n2))
         for i in range(rank)], axis=0)
    return estimate
