%% configurations

% experiment configurations
n1 = 50;
n2 = 60;
r = 3;
condition_number = 1e1;  % condition number
oversampling_ratio = 1.5;
singular_values = linspace(1, condition_number, r);

% algorithm options (for more options, see GNMR_sensing.m)
clear opts
opts.verbose = 1;               % display intermediate results
opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
% number of iterations
opts.max_outer_iter = 100;      % maximal number of outer iterations
opts.max_inner_iter = 2000;     % maximal number of inner iterations for the LSQR solver
% stopping criteria (-1 to disable a criterion)
opts.stop_relRes = 1e-14;   	% small relRes threshold
                                % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
opts.stop_relDiff = 1e-14;      % small relative X_hat difference threshold


%% run experiment
format long;
fprintf('\n n1,n2: %4d,%4d. rank: %2d. condition number: %e \n oversampling ratio: %e\n\n', ...
    n1, n2, r, condition_number, oversampling_ratio);

rng_value = 2021;
rng('default');
rng(rng_value);

% generate low rank matrix X0
[X0, U0, V0] = generate_matrix(n1,n2,singular_values);

% generate sensing operator of Gaussian measurements
m = ceil(oversampling_ratio * r * (n2+n1-r));  % number of observations
A = normrnd(0, 1.0/sqrt(m), m, n1*n2);

% compute b, the observed linear measurements, according to A
X0_vec = X0(:);
b = A * X0_vec;

% run GNMR
[X_hat, ~, iter] = GNMR_sensing(b, A, n1, n2, r, opts);

% report
true_error = norm(X_hat - X0, 'fro') / norm(X0, 'fro');
fprintf('true error %8d, iter %3d\n', true_error, iter);

