clear;
close all;
%% Constants
filename = 'train-images.idx3-ubyte';
label_filename = 'train-labels.idx1-ubyte';
test_images_filename = 't10k-images.idx3-ubyte';
test_labels_filename = 't10k-labels.idx1-ubyte';

test_cases = 1;
from = 0;
samples = 60000;
%% Load images and labels
images = loadMNISTImages(filename, samples, 0);
labels = loadMNISTLabels(label_filename, samples, 0);
test_images = loadMNISTImages(test_images_filename, test_cases, from);
test_labels = loadMNISTLabels(test_labels_filename, test_cases, from);

features = [784, 300, 100, 20, 5];
        
for i = 1:5
    %% find eigendigits 
    [x, k] = size(images);
    [m, V] = hw1FindEigendigits(images);
    V = V(:, 1:features(i));
    % project test data onto eigenspace
    reduced_testing_set_n = (test_images)' * V;
    figure(i);
    imshow(reshape(V*reduced_testing_set_n', 28, 28));
end