function [X, U, V] = generate_matrix(n1,n2,singular_values)
%
% Code taken from R2RILS by Jonathan Bauch, Boaz Nadler and Pini Zilber
%
% INPUT: n_r,n_c = number of rows and columns
% lambda: non-zero singular values

% OUTPUT: X = n1 x n2 matrix of rank r
%         U = n1 x r left singular vectors
%         V = n2 x r right singular vectors

r = length(singular_values); 
D = diag(singular_values);    % diagonal r x r matrix
Z = randn(n1,r); 

[U, ~, ~] = svd(Z,'econ'); 

Z = randn(n2,r); 
[V, ~, ~] = svd(Z,'econ'); 

X = U * D * V';