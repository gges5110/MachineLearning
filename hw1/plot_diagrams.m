%% Plot for the result of accuracy against features.
accuracy1 = [0.2491, 0.3898, 0.4428, 0.5738, 0.6937, 0.7889, 0.8430, 0.8756, 0.8941, 0.9137, 0.9491, 0.9637, 0.9718, 0.9735, 0.9734, 0.9729, 0.9716, 0.9691, 0.9688, 0.9691];
features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 70, 100, 200, 300, 784];
figure(1);
plot(features, accuracy1)
xlabel('Features');
ylabel('Accuracy');
%% Plot for the result of accuracy against the size of training set.
accuracy2 = [0.3610, 0.5490, 0.6411, 0.8057, 0.8459, 0.9169, 0.9363, 0.9595, 0.9637,];
samples = [6, 30, 60, 300, 600, 3000, 6000, 30000, 60000];
figure(2);
plot(samples, accuracy2)
xlabel('Training Data');
ylabel('Accuracy');