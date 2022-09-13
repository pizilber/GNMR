# GNMR
This repository contains `Matlab` and `Python` implementations for `GNMR` (Gauss-Newton for Matrix Recovery), as described in [P. Zilber and B. Nadler (SIMODS 2022)](https://epubs.siam.org/doi/abs/10.1137/21M1433812) or in the [arXiv version](http://arxiv.org/abs/2106.12933), both in the matrix completion and matrix sensing cases.

## Usage
Simple demos demonstrating the usage of `GNMR` are available, see `LRMC_demo.m`/`LRMC_demo.py` for matrix completion and `LRMS_demo.m`/`LRMS_demo.py` for matrix sensing.
In these demos, the expected configurations of the matrix to be recovered are:
- n1: number of rows
- n2: number of columns
- rank
- condition number
- oversampling ratio: number of observed entries, normalized by the factor _(n1 + n2 - rank)*rank_. Oversampling ratio of one corresponds to the information limit

Given these configurations, the demo generates a matrix using `generate_matrix.m`/`generate_matrix.py`, and a mask (sampling pattern) in case of matrix comletion using `generate_mask.m`/`generate_mask.py`, and then run the `GNMR` algorithm.
The basic configurations of `GNMR` are:
- opts.verbose: display intermediate result
- opts.alpha: the variant of `GNMR` (e.g. alpha=1 is the setting variant)
- opts.max_outer_iter: maximal number of outer iterations
- opts.max_inner_iter: maximal number of inner iterations of the LSQR solver
- opts.stop_relRes: early stop if the rel-RMSE distnace from the underlying matrix is less than this threshold (-1 to diasable)
- opts.stop_relDiff: early stop if the estimate does not change by a factor of this threshold (-1 to disable)

Additional (optional) configurations can be found in the beginning of the files `GNMR_completion.m`/`GNMR_completion.py` and `GNMR_sensing.m`/`GNMR_sensing.py`.

Note that unlike several other matrix recovery algorithms, for `GNMR` the _opts.max_inner_iter_ parameter is usually more important than _opts.max_outer_iter_.
The required number of inner iterations depends on the dimension and the difficulty of the problem (the condition number and the oversampling ratio), but to ensure recovery a few thousand is generically recommended.

## Citation
If you refer to the method or the paper, please cite them as:
```
@article{zilber2022gnmr,
  title={GNMR: A provable one-line algorithm for low rank matrix recovery},
  author={Zilber, Pini and Nadler, Boaz},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={4},
  number={2},
  pages={909--934},
  year={2022},
  publisher={SIAM}
}
```
