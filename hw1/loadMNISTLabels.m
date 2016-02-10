function labels = loadMNISTLabels(filename, num, from)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, num + from, 'unsigned char');
labels = labels(from + 1:num + from);
%assert(size(labels,1) == num, 'Mismatch in label count');

fclose(fp);

end
