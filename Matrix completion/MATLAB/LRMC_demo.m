%% configurations

% experiment configurations
n1 = 300;
n2 = 350;
r = 5;
condition_number = 1e1;
oversampling_ratio = 1.5;
singluar_values = linspace(1, condition_number, r);
remask_repeat = 1e8; % maximal attempts at generating a mask
                        % with r observed entries in columns and rows

% algorithm options (for more options, see GNMR_completion.m)
clear opts
opts.verbose = 1;               % display intermediate results
opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
% number of iterations
opts.max_outer_iter = 100;      % maximal number of outer iterations
opts.max_inner_iter = 2000;     % maximal number of inner iterations for the LSQR solver
% early stopping criteria (-1 for disabling criterion)
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
[X0, U0, V0] = generate_matrix(n1,n2,singluar_values);

% generate mask
m = min(floor(r*(n2+n1-r) * oversampling_ratio), n1*n2); % number of observed entries
[H, omega, omega_2d, mask_flag] = generate_mask(n1,n2,m,r,remask_repeat);
if mask_flag
    disp('mask found');
else
    disp('mask was not found, exiting...');
    return;
end

% compute X, the observed matrix
X = sparse(omega_2d(:,1),omega_2d(:,2),X0(omega),n1,n2);

% run GNMR
[X_hat, ~, iter] = GNMR_completion(X, omega_2d, r, opts);

% report
true_error = norm(X_hat - X0, 'fro') / norm(X, 'fro');
fprintf('true error %8d, iter %3d\n', true_error, iter);
