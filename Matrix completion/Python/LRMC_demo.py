import numpy as np
import time
from generate_matrix import generate_matrix
from generate_mask import generate_mask
from GNMR_completion import GNMR_completion,\
    INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED

def run_demo():
    # experiment configurations
    n1 = 300
    n2 = 350
    rank = 5
    condition_number = 1e1
    oversampling_ratio = 1.5
    singular_values = np.linspace(1, condition_number, rank)
    print("n1, n2:", n1, n2)
    print("oversampling ratio:", oversampling_ratio)
    print("singular values:", singular_values)

    # algorithm options (for documentation and more options, see GNMR_completion.py)
    options = {
        # general
        'verbose' : True,
        'alpha' : 1,  # (1: setting variant, 0: averaging variant, -1: updating variant)
        # number of iterations
        'max_outer_iter' : 100, 
        'max_inner_iter': 2000,
        # early stopping criteria (-1 to disable a criterion)
        'stop_relRes':  5e-14,
        'stop_relDiff': 5e-14,
    }

    # calculate full matrix, mask and corresponding observed matrix
    p = oversampling_ratio * rank * (n1 + n2 - rank) / (n1 * n2)
    X0 = generate_matrix(n1, n2, singular_values)
    omega = generate_mask(n1, n2, rank, p)
    X = X0 * omega

    # run GNMR
    start = time.time()
    X_hat, iter, _, _ = GNMR_completion(X, omega, rank, **options)
    end = time.time()

    # report
    print("iter: ", iter, ". elapsed time:", end - start)
    true_error = np.linalg.norm(X_hat - X0, ord='fro') / np.linalg.norm(X0, ord='fro')
    observed_error = np.linalg.norm((X_hat - X0) * omega, ord='fro') / np.linalg.norm(X0, ord='fro')
    print('true error: {}, observed error: {}'.format(true_error, observed_error))


if __name__ == '__main__':
    run_demo()
