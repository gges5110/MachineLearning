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
features = 50;

% For test data
test_cases = 10000;
from = 0;

samples_set = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000];
accuracy_samples = zeros(1, size(samples_set, 2));

for samples = 1:size(samples_set, 2)
    %% =============== Part 2: Loading Data ================
    Tr = loadMNISTImages(filename, samples_set(samples), 0);
    labels = loadMNISTLabels(label_filename, samples_set(samples), 0);
    fprintf('Finish loading training data.\n');

    % load test data
    Ts = loadMNISTImages(test_images_filename, test_cases, from);
    test_labels = loadMNISTLabels(test_labels_filename, test_cases, from);
    fprintf('Finish loading test data.\n');

    Total_T = zeros(size(Tr, 1), size(Tr, 2) + size(Ts, 2));
    Total_T(:, 1:size(Tr, 2)) = Tr;
    Total_T(:, (1:size(Ts, 2)) + size(Tr, 2)) = Ts;

    %% =============== Part 3: Find Eigendigits and Scaling data ================
    [x, k] = size(Total_T);
    [m, V_original] = findEigendigits(Total_T);
    V = V_original(:, 1:features);
    fprintf('PCA done.\n');

    Total_T = Total_T'*V;
    Total_T = zscore(Total_T);

    Tr_z = Total_T(1:samples_set(samples), :);
    Ts_z = Total_T((1:test_cases) + samples_set(samples), :);
    fprintf('Finish projecting and scalaing.\n');
    %% ==================== Part 4: Training Linear SVM ====================
    cmd = '-q -c 64 -g 0.03125';
    % cmd = ['-q -c ', num2str(2^2), ' -g ', num2str(2^-12)];
    Mdl = svmtrain(labels, Tr_z, cmd);
    fprintf('Finish training SVM model.\n');

    %% ==================== Part 5: Predicting Data Using SVM ====================
    % Predict using the SVM model
    [predicted_label, accuracy, prob_values] = svmpredict(test_labels, Ts_z, Mdl);
    fprintf('Finish prediction.\n');
    accuracy_samples(samples) = accuracy(1);
end
