close all

mu = [-25, 10];
sigma = [10, 10];
low_b = -50;
high_b = 50;
x = [low_b:.1:high_b];
norm = 0.3*normpdf(x, mu(1), sigma(1)) + 0.7*normpdf(x, mu(2), sigma(2));

M = 3;
candidate_density = @(x) normpdf(x, 10, 30);
% candidate_density = @(x) 0.03;
figure(1);
hold on;
plot(x, norm);
plot(x, M * candidate_density(x));
hold off;

unknown_density = @(x) 0.3*normpdf(x, mu(1), sigma(1)) + 0.7*normpdf(x, mu(2), sigma(2));
proposal_density = @(x, y) normpdf(x, y, 30);
%% Sampling using rejection sampling
sampleSize = 50000;
samples = zeros(1, sampleSize);
i = 1;

while i <= sampleSize
    xc = rand*(high_b - low_b) + low_b;
    accept = candidate_density(xc) / (M * unknown_density(xc));
    u = rand;
    if (accept < u)
        samples(i) = xc;
        i = i + 1;
    end
end

figure(3);
histfit(samples);

sample = zeros(high_b - low_b + 1, 1);
%% Discretize the samples
for i = 1:sampleSize
    sample(floor(samples(i)) - low_b + 1) = sample(floor(samples(i)) - low_b + 1) + 1;
end

figure(2);
% histfit()
bar(low_b:high_b, sample);
