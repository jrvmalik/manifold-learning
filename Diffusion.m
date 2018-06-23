function [ E, S ] = Diffusion( data, r, d, flag )
% Calculate the time-1 diffusion map for a pre-normalized data matrix.
%   INPUT
%       data : Data matrix (n x p).
%       r    : Number of neighbours (e.g. 15).
%       d    : Dimension of embedding (e.g. 8).
%   OUTPUT
%       E    : Diffusion embedding (n x d).
%       S    : Singular values (d + 1 x d + 1).
% Written by John Malik on 2018.6.23, john.malik@duke.edu.

if nargin < 4
    flag = false;
end

% number of points
n = size(data, 1);

% neighbour search
jj = knnsearch(data, data, 'k', r);
ii = transpose(1:n) * ones(1, r);

% adjacency matrix
A = sparse(ii, jj, ones(size(ii)), n, n);

% affinity matrix
if ~flag
    % number of neighbours x and y have in common
    W = A * A';
else W = A' * A; % number of pairs with x and y as common neighbours
end

% alpha normalize
D = sum(W, 2);
W = bsxfun(@rdivide, bsxfun(@rdivide, W, D), transpose(D));

% normalized graph laplacian
D = sqrt(sum(W, 2));
P = bsxfun(@rdivide, bsxfun(@rdivide, W, D), transpose(D));

% eigendecomposition
opts.issym = 1;
opts.isreal = 1;
opts.v0 = ones(n, 1);
[E, S] = eigs((P + P') / 2, d + 1, 'la', opts);

% shift back and normalize
E = bsxfun(@rdivide, E(:, 2:end), D);
E = E * S(2:end, 2:end);

end

