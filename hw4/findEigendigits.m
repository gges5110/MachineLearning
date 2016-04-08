function [m, V] = hw1FindEigendigits(A)
%% Definition
% Matrix A = (x by k), x is the total number of pixels in an image and k is the number of training images
% vector m = (x by 1), containing the mean column vector of A
% Matrix V = (x by k), contains k eigenvectors of the covariance matrix of A 
[x, k] = size(A);
%% Mean Column Vector
m = sum(A, 2) / k;
%% Eigenvector of Covariance Matrix of A 
% Subtract the covariance matrix by mean
A_sub = A - m * ones(1, k);
% First find x by x matrix
C = cov(A_sub');
% svd will give descending order normalized eigenvectors
[V, ~, ~] = svd(C);
end