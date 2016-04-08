function [ grad ] = gradient( eta, Y, W )
Z = sigmoid(Y);
grad = eta*(eye(size(Y, 1)) + (1 - 2*Z)*Y')*W;
end

