close all

mu = [-25, 10];             % Target PDF means
sigma = [10, 10];           % Target PDF varicanes
low_b = -50; high_b = 50;
x = low_b:.1:high_b;

pdf = @(x) 0.3*normpdf(x, mu(1), sigma(1)) + 0.7*normpdf(x, mu(2), sigma(2));   % Target PDF
variance = 1;                                                                  % Proposal PDF variance            
proppdf = @(x,y) normpdf(x, y, variance);                                       % Proposal PDF
proprnd = @(x) normrnd(x, variance);                                            % Random distribution from proposal PDF

%% Sampling using Metropolis-Hasting sampling
sampleSize = 30000;
theta = zeros(1, sampleSize);
theta(1) = 5;

for i = 1:sampleSize
    theta_ast = proprnd(theta(i));    % Sampling from proposal PDF
    r = pdf(theta_ast)/pdf(theta(i));
    if rand <= min(1, r)
       theta(i + 1) = theta_ast;
    else 
        theta(i + 1) = theta(i);
    end    
end

sample = zeros(floor(max(theta)) - floor(min(theta)) + 1, 1);
%% Discretize the samples
for i = 1:sampleSize
    index = floor(theta(i)) - floor(min(theta)) + 1; 
    sample(index) = sample(index) + 1;
end

figure(1);
hold on;
bar(floor(min(theta)):floor(max(theta)), sample/sampleSize,'FaceColor',[.8 .8 1]);
plot(floor(min(theta)):floor(max(theta)), pdf(floor(min(theta)):floor(max(theta))), 'r', 'LineWidth',2);

