close all

%
mu = [-25, 10];             % Target PDF means
sigma = [10, 10];           % Target PDF varicanes
low_b = -50; high_b = 50;
x = low_b:.1:high_b;
delta = high_b - low_b;

pdf = @(x) 0.3*normpdf(x, mu(1), sigma(1)) + 0.7*normpdf(x, mu(2), sigma(2));   % Target PDF
variance = 10;                                                                  % Proposal PDF variance            
proppdf = @(x,y) unifpdf(y-x,low_b,high_b);                                     % Proposal PDF
proprnd = @(x) x + rand*2*delta - delta;                                        % Random distribution from proposal PDF


%% Sampling using Metropolis-Hasting sampling
sampleSize = 50000;
theta = zeros(1, sampleSize);
theta(1) = 0.3;

for i = 1:sampleSize
    theta_ast = proprnd([]);    % Sampling from proposal PDF
    r = pdf(theta_ast)/pdf(theta(i));
    if rand <= min(1, r)
       theta(i + 1) = theta_ast;
    else 
        theta(i + 1) = theta(i);
    end    
end

sample = zeros(high_b - low_b + 1, 1);
%% Discretize the samples
for i = 1:sampleSize
    sample(floor(samples(i)) - low_b + 1) = sample(floor(samples(i)) - low_b + 1) + 1;
end

figure(2);
% histfit()
bar(low_b:high_b, sample);
