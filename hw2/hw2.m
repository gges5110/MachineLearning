clear; close all;
%% 1.	Get data. 
filename = 'sounds.mat';
load(filename);
% A is the data mixing matrix, U is the source signals 
U = sounds;
[numSrc, numData] = size(U);
A=rand(numSrc, numSrc);
[m, n] = size(A);

plot(0,0);
hold on;
xlabel('Time');
offSet=-1;
offSet = addtoPlot(U, offSet, 'src'); 
%% 2.	Mix the data. 
X = A*U;
% subtract the mean
M = sum(X,2) / size(X,2);
X = X - M*ones(1, size(X,2));
offSet = addtoPlot(X, offSet, 'mix'); 
%% 3.	Algorithm.
% Initialize W
iterations = 10000;
eta0 = 0.01; % learning rate
T = 1000;
eta = eta0;
W = rand(size(A)) ./ 10;

for i = 1: iterations    
    del_W = gradient(eta, W * X, W);    % gradient descent - shift by delta
    W = W + del_W;                      % update W
    eta = eta0 / (1 + (i/T));           % annealing - learning rate
    if sum(sum(W)) > 20
       break; 
    end
end
%% 4.   Test. 
Y = W * X;
Y = (Y - min(min(Y))) ./ (max(max(Y)) - min(min(Y)));
offSet = addtoPlot(Y, offSet, 'rec'); 