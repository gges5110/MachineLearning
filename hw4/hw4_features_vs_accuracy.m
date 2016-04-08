%% Initialization
clear ; close all; clc

% addpath to the libsvm toolbox
addpath('../libsvm/matlab');

%% =============== Part 1: Define Parameters ================
filename = 'train-images.idx3-ubyte';
label_filename = 'train-labels.idx1-ubyte';
test_images_filename = 't10k-images.idx3-ubyte';
test_labels_filename = 't10k-labels.idx1-ubyte';

% For training data
samples = 5000;
features = 60;

% For test data
test_cases = 10000;
from = 0;

%% =============== Part 2: Loading Data ================
Tr = loadMNISTImages(filename, samples, 0);
labels = loadMNISTLabels(label_filename, samples, 0);
fprintf('Finish loading training data.\n');

%% =============== Part 3: Find Eigendigits  ================
[x, k] = size(Tr);
[m, V_original] = findEigendigits(Tr);

features_to_test = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 784];
accuracy_total = zeros(1, size(features_to_test, 2));

for features = 1:size(features_to_test, 2)
    V = V_original(:, 1:features_to_test(features));
    fprintf('PCA done.\n');

    %% ==================== Part 4: Training Linear SVM ====================
    Tr_trans = (Tr - m * ones(1, size(Tr, 2)))' * V;

    Mdl = svmtrain(labels, Tr_trans, '-q -c 64 -g 0.03125');
    fprintf('Finish training SVM model.\n');

    %% ==================== Part 5: Predicting Data Using SVM ====================
    % load test data
    Ts = loadMNISTImages(test_images_filename, test_cases, from);
    test_labels = loadMNISTLabels(test_labels_filename, test_cases, from);
    fprintf('Finish loading test data.\n');

    % project test data onto eigenspace
    Ts_trans = (Ts - m * ones(1, size(Ts, 2)))' * V;

    % Predict using the SVM model
    [predicted_label, accuracy, prob_values] = svmpredict(test_labels, Ts_trans, Mdl);
    fprintf('Finish prediction.\n');
    accuracy_total(features) = accuracy(1);
end
