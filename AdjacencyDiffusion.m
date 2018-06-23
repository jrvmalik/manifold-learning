function [ E, S ] = AdjacencyDiffusion( data, r, d )
% Calculate the time-1 diffusion map for a pre-normalized data matrix.
% Use the zero-one kernel.
%   INPUT
%       data : Data matrix (n x p).
%       r    : Number of neighbours (e.g. 15).
%       d    : Dimension of embedding (e.g. 8).
%   OUTPUT
%       E    : Diffusion embedding (n x d).
%       S    : Singular values (d + 1 x d + 1).
% Written by John Malik on 2018.6.23, john.malik@duke.edu.

% number of points
n = size(data, 1);

% neighbour search
jj = knnsearch(data, data, 'k', r);
ii = transpose(1:n) * ones(1, r);

% affinity matrix
ker = ones(size(ii));
W = sparse(ii, jj, ker, n, n);
W = max(W, W');

% alpha normalize
D = sum(W, 2);
W = bsxfun(@rdivide, bsxfun(@rdivide, W, D), transpose(D));

% random walk
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

