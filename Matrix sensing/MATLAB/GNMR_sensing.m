function [X_hat, all_relRes, iter, convergence_flag] = ...
    GNMR_sensing(b,A,n1,n2,r,opts)
%
% Written by Pini Zilber & Boaz Nadler / 2021
% based on the code of R2RILS by Jonatahan Bauch, Boaz Nadler and Pini Zilber
% with modifications by Christian Kuemmerle (sparse matrix construction of A)
%
% INPUT: 
% b = observed linear measurements of the underlying matrix
% A = the sensing operator
% n1 = number of rows in the underlying matrix
% n2 = number of columns in the underlying matrix
% r = target rank of the underlying matrix
% opts = options meta-variable (see opts_default for details)
%
% OUTPUT:
% X_hat = rank-r approximation of the underlying matrix
% all_relRes = list of the residual errors of the least-squares problem throughout the iterations
% iter = final iteration number
% convergence_flag = indicating whether the algorithm converged
%

%% configurations: default values for option variables
opts_default.verbose = 1;                       % display intermediate results
opts_default.alpha = 1;                         % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
% number of iterations and tolerance
opts_default.max_outer_iter = 100;              % maximal number of outer iterations
opts_default.max_inner_iter = 2000;             % maximal number of inner iterations for the LSQR solver
opts_default.inner_init_tol = 1e-15;            % initial tolerance of the LSQR solver
opts_default.LSQR_smart_tol = 1;                % use LSQR_tol==relRes^2 when relRes is low enough
opts_default.LSQR_smart_obj_min = 1e-5;         % relRes threshold to start using LSQR smart tol
% initialization
opts_default.init_option = 0;                   % 0 for SVD, 1 for random, 2 for opts.init_U, opts.init_V
opts_default.init_U = NaN;                      % if opts.init_option==2, use this initialization for U
opts_default.init_V = NaN;                      % if opts.init_option==2, use this initialization for V
% early stopping criteria (-1 to disable a criterion)
opts_default.stop_relRes = 1e-16;               % small relRes threshold (relevant to noise-free case)
opts_default.stop_relDiff = -1;                 % small relative X_hat difference threshold
opts_default.stop_relResDiff = -1;              % small relRes difference threshold
opts_default.stop_relResStuck_ratio = -1;       % stop if minimal relRes didn't change by a factor of stop_relResNoChange_ratio...
opts_default.stop_relResStuck_iters = -1;       % ... in the last #stop_relResNoChange_iters outer iterations
% additional configurations
opts_default.r_projection_in_iteration = 0;     % if true, error estimation at each iteration is calculated
                                                % for the best rank-r approximation using SVD

% for each unset option set its default value
fn = fieldnames(opts_default);
for k=1:numel(fn)
    if ~isfield(opts,fn{k}) || isempty(opts.(fn{k}))
        opts.(fn{k}) = opts_default.(fn{k});
        fn{k} = opts_default.(fn{k});
    end
end

%% some definitions
[omega1, omega2] = ndgrid(1:n1,1:n2);
omega = [omega1(:), omega2(:)];  % list of all possible pairs in [n1] x [n2]
colind_A = generate_sparse_matrix_indices(omega,r,n2);

%% initialize U and V (of sizes n1 x r and n2 x r)
if opts.init_option == 0
    % initialization by A' * b
    X = A' * b;
    X = reshape(X, n1, n2);
    [U, ~, V] = svds(X,r);
elseif opts.init_option == 1
    % initialization by random orthogonal matrices
    Z = randn(n1,r);
    [U, ~, ~] = svd(Z,'econ'); 
    Z = randn(n2,r);
    [V, ~, ~] = svd(Z,'econ'); 
else
    % initiazliation by user-defined matrices
    U = opts.U_init;
    V = opts.V_init; 
end

%% before iterations
X_hat_previous = U*V';   % stores rank-r projection of previous intermediate rank 2r estimate
all_relRes = zeros(opts.max_outer_iter,1);
relRes = Inf;
best_relRes = Inf;
convergence_flag = 0;

%% iterations
for iter = 1:opts.max_outer_iter
    
    %% construct variables for LSQR
    L = generate_sparse_L(U,V,omega,colind_A);
    
    % update rhs (b in ||Ax-b||^2)
    UV_vec = U*V';
    UV_vec = UV_vec(:);
    b_t = b + opts.alpha * A * UV_vec;

    %% solve LSQR
    % determine tolerance for LSQR solver
    LSQR_tol = opts.inner_init_tol;
    if opts.LSQR_smart_tol && relRes < opts.LSQR_smart_obj_min
        LSQR_tol = min(LSQR_tol, relRes^2);
    end
    LSQR_tol = max(LSQR_tol, 2*eps);  % to supress warning
    % solve the least squares problem
    [z, ~, relRes, LSQR_iters_done] = lsqr(A * L, b_t, LSQR_tol, opts.max_inner_iter);
        % LSQR finds the minimum norm solution and is much faster than lsqminnorm
    relRes = relRes * (norm(b_t) / norm(b));
        % original relRes is normalized w.r.t. to b_t,
        % but we want to normalize w.r.t. b

    %% construct Utilde and Vtilde from the solution Z
    Utilde = zeros(size(U));
    Vtilde = zeros(size(V)); 
    nc_list = r * [0:1:(n2-1)]; 
    for i=1:r
        Vtilde(:,i) = z(i+nc_list); 
    end
    nr_list = r * [0:1:(n1-1)]; 
    start_idx = r*n2; 
    for i=1:r
        Utilde(:,i) = z(start_idx + i + nr_list);
    end
    
    %% calculate U_next, V_next
    U_next = 0.5*(1 - opts.alpha) * U + Utilde;
    V_next = 0.5*(1 - opts.alpha) * V + Vtilde;
    
    %% get new estimate
    % intermediate rank-2r estimate
    X_hat_2r = U * V_next' + U_next * V' - U * V';
    
    % rank r projection of intermediate rank-2r solution
    if opts.r_projection_in_iteration
        [U_r, Sigma_r, V_r] = svds(X_hat_2r,r); 
        X_hat = U_r * Sigma_r * V_r';
    else
        X_hat = U_next * V_next';
    end
    
    % store relRes and update X_hat_2r_best if needed
    all_relRes(iter) = relRes; 
    if relRes < best_relRes
        best_relRes = relRes;
        X_hat_2r_best = X_hat_2r;
    end
    
    % update U, V and X_hat_diff
    U = U_next;
    V = V_next;
    X_hat_diff = norm(X_hat - X_hat_previous, 'fro') / norm(X_hat, 'fro');
    
    %% report
    if opts.verbose
        fprintf('[INSIDE GNMR] iter %4d \t diff X_r %5d\t relRes %6d\n',...
            iter, X_hat_diff, relRes);
    end
    
    %% check early stopping criteria
    if relRes < opts.stop_relRes
        msg = '[INSIDE GNMR] Early stopping: small error on observed entries\n';
        convergence_flag = 1;
    elseif X_hat_diff < opts.stop_relDiff
        msg = '[INSIDE GNMR] Early stopping: X_hat does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && ...
            abs(all_relRes(iter-1)/relRes-1) < opts.stop_relResDiff
        msg = '[INSIDE GNMR] Early stopping: relRes does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && LSQR_iters_done == 0
        msg = '[INSIDE GNMR] Early stopping: no iterations of LSQR solver\n';
        convergence_flag = 1;
    elseif iter >= 3 * opts.stop_relResStuck_iters && opts.stop_relResStuck_ratio > 0 ...
            && mod(iter, opts.stop_relResStuck_iters) == 0
        % the factor of 3 is since we want to ignore the first opts.stop_relResStuck_iters
        % iteratios, as they may contain small observed relRes with large unobserved...
        last_relRes = all_relRes(iter-opts.stop_relResStuck_iters+1:iter);
        previous_relRes = all_relRes(iter-2*opts.stop_relResStuck_iters+1:iter-opts.stop_relResStuck_iters);
        if min(last_relRes) / min(previous_relRes) > opts.stop_relResStuck_ratio
            msg = sprintf('[INSIDE GNMR] Early stopping: relRes did not decrease by a factor of %f in %d iterations\n', ...
                opts.stop_relResStuck_ratio, opts.stop_relResStuck_iters);
            convergence_flag = 1;
        end
    end
    if convergence_flag
        if opts.verbose
            fprintf(msg);
        end
        break
    end
    
    %% update before next iterate
    X_hat_previous = X_hat;
end

%% return
[U_hat, lambda_hat, V_hat] = svds(X_hat_2r_best,r);
X_hat = U_hat * lambda_hat * V_hat';
end

%% auxiliary functions for generating sparse matrix A
% taken from the matrix completion algorithm,
% can probably be optimized for matrix sensing

function L = generate_sparse_L(U,V,omega,colind_L)
nv = size(omega,1);
r = size(U,2);
n1 = size(U,1);
n2 = size(V,1);
%L = zeros(nv,m);   % matrix of least squares problem 
val_L=zeros(1,2*r*nv);
rowind_L=kron(1:nv,ones(1,2*r));
for counter=1:nv
    j = omega(counter,1); k = omega(counter,2);
    %colind_L = [colind_L,(r*(k-1)+1):(r*(k-1)+r),(r*nc + r*(j-1)+1):(r*nc + r*(j-1)+r)];
    val_L((counter-1)*(2*r)+1:counter*2*r) = [U(j,:),V(k,:)];
end
L = sparse(rowind_L,colind_L,val_L,nv,r*(n1+n2));
end

function colind_L = generate_sparse_matrix_indices(omega,r,n2)
nv = length(omega);
colind_L=zeros(1,2*r*nv);
for counter=1:nv
    j = omega(counter,1); k = omega(counter,2);
    colind_L((counter-1)*(2*r)+1:counter*2*r) = [(r*(k-1)+1):(r*(k-1)+r),...
        (r*n2 + r*(j-1)+1):(r*n2 + r*(j-1)+r)];
end
end
