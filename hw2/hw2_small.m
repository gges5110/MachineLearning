%% 1.	Get data. 
filename = 'icaTest.mat';
load(filename);
% A is the data mixing matrix, U is the source signals 
%% 2.	Mix the data. 
X = A*U;
M = sum(X,2) / size(X,2);
X = X - M*ones(1, size(X,2));
%% 3.	Algorithm.
% Initialize W
W = random('unif', 0, 0.1, 3, 3);
iterations = 1000000;
eta = 0.01; % learning rate
for i = 1: iterations
    Y = W * X;
    Z = sigmoid(Y);
    del_W = eta*(eye(size(Y, 1)) + (1 - 2*Z)*Y')*W;
    W = W + del_W;
end
%% 4.	Test. 
figure(1);
subplot(3,1,1);
plot(U(1, :));

subplot(3,1,2);
plot(U(2, :));

subplot(3,1,3);
plot(U(3, :));


Y = W * X;
figure(2);
subplot(3,1,1);
plot(Y(1, :));

subplot(3,1,2);
plot(Y(2, :));

subplot(3,1,3);
plot(Y(3, :));