import numpy as np
import time
from generate_matrix import generate_matrix
from GNMR_sensing import GNMR_sensing,\
    INIT_WITH_ATb, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED

def run_demo():
    # experiment configurations
    n1 = 50
    n2 = 60
    rank = 3
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

    # calculate full matrix, measurement operator and corresponding observed vector
    p = oversampling_ratio * rank * (n1 + n2 - rank) / (n1 * n2)
    X0 = generate_matrix(n1, n2, singular_values)
    m = int(np.ceil(oversampling_ratio * rank * (n1+n2-rank)))
    A = np.random.normal(0, 1.0/np.sqrt(m), (m, n1*n2))
    b = A @ np.ravel(X0)

    # run GNMR
    start = time.time()
    X_hat, iter, _, _ = GNMR_sensing(b, A, n1, n2, rank, **options)
    end = time.time()

    # report
    print("iter: ", iter, ". elapsed time:", end - start)
    true_error = np.linalg.norm(X_hat - X0, ord='fro') / np.linalg.norm(X0, ord='fro')
    observed_error = np.linalg.norm((A @ np.ravel(X_hat) - b)) / np.linalg.norm(b)
    print('true error: {}, observed error: {}'.format(true_error, observed_error))


if __name__ == '__main__':
    run_demo()
