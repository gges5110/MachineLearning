clear;
%% Constants
filename = 'train-images.idx3-ubyte';
label_filename = 'train-labels.idx1-ubyte';
test_images_filename = 't10k-images.idx3-ubyte';
test_labels_filename = 't10k-labels.idx1-ubyte';
% For training data
samples = 30000;
features = 20;
% For test data
test_cases = 10000;
from = 0;
%% Load images and labels
Tr = loadMNISTImages(filename, samples, 0);
labels = loadMNISTLabels(label_filename, samples, 0);
fprintf('Finish loading training data.\n');
%% find eigendigits 
[x, k] = size(Tr);
[m, V] = hw1FindEigendigits(Tr);
V = V(:, 1:features);
fprintf('PCA done.\n');
%% Use KNN to classify
% Dimension reduced training set
Tr_trans = (Tr - m * ones(1, k))' * V;
Mdl = fitcknn(Tr_trans, labels, 'NumNeighbors', 1);
fprintf('Finish training knn model.\n');
%% Test
% load test data
Ts = loadMNISTImages(test_images_filename, test_cases, from);
test_labels = loadMNISTLabels(test_labels_filename, test_cases, from);
fprintf('Finish loading test data.\n');
% project test data onto eigenspace
Ts_trans = (Ts - m * ones(1, test_cases))' * V;
% Predict using the KNN model
predicted_label = predict(Mdl, Ts_trans);
fprintf('Finish prediction.\n');
%% Calculate accuracy
tot = size(Ts, 2);
correct = sum(predicted_label == test_labels);
accuracy = correct / tot