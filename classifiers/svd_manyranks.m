% Note: This code uses svd once instead svds many times to be faster when
% searching for the best order of approximation to use 
close all
clear all

% User Parameters
DATASET = 'usps';
VALIDATION_SIZE = .2;  % integer of decimal (percentage)
TESTING_SIZE = .2;  % integer of decimal (percentage)
RANKS2TRY = 'all';  % must be 'all' or list of integers

datadir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
if lower(DATASET) == 'usps'
    load(fullfile(datadir, 'usps', 'USPS.mat'))
elseif lower(DATASET) == 'mnist'
	load(fullfile(datadir, 'mnist', 'MNIST.mat'))
elseif lower(DATASET) == 'notmnist_small'
    load(fullfile(datadir, 'notMNIST', 'notMNIST_small_no_duplicates.mat'))
elseif lower(DATASET) == 'notmnist_large'
    load(fullfile(datadir, 'notMNIST', 'notMNIST_large_no_duplicates.mat'))
else
    dataset_dict = loadmat(DATASET)
end

[m h w] = size(X);
n = w*h;
X = reshape(X, m, n);

% Let's shuffle
shuffle = randperm(m);
X = X(shuffle, :);
y = y(shuffle, :);

% parse ranks2try
if RANKS2TRY == 'all'
	ranks2try = 1:rank(X);
else
	ranks2try = RANKS2TRY
end

% define some variables for convenience
distinct_labels = unique(y)';

% Let's take 10k of the training examples to serve as a
% validation set and another 10k for a testing set
if VALIDATION_SIZE < 1
	m_valid = round(VALIDATION_SIZE*m);
else
	m_valid = VALIDATION_SIZE
end
if TESTING_SIZE < 1
	m_test = round(TESTING_SIZE*m);
else
	m_test = TESTING_SIZE
end

m_train = m - m_valid - m_test;
X_valid = X(1:m_valid, :);
y_valid = y(1:m_valid, :);
X_test = X(m_valid+1: m_valid+m_test, :);
y_test = y(m_valid+1: m_valid+m_test, :);
X_train = X(m_valid+m_test+1: end, :);
y_train = y(m_valid+m_test+1: end, :);

% normalize data
% X = X - mean(X);  % hurts accuracy
% X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

printf('\nCheck Balance:\n')
printf('Totals -- ')
for l=distinct_labels
	printf([num2str(sum(y==l)) ' '])
end
printf('\nTraining Set -- ')
for l=distinct_labels
	printf([num2str(sum(y_train==l)) ' '])
end
printf('\nValidation Set -- ')
for l=distinct_labels
	printf([num2str(sum(y_valid==l)) ' '])
end
printf('\nTesting Set -- ')
for l=distinct_labels
	printf([num2str(sum(y_test==l)) ' '])
end

% Convert to sparse arrays, if it helps speed
tic
% X_train = sparse(X_train);
% X_valid = sparse(X_valid);
% X_test = sparse(X_test);

printf('\n\nTraining & Validation Stage:\n')
V = NaN(n, n, length(distinct_labels));
for l=distinct_labels
	[u s V(:,:,l+1)] = svd(X_train( find(y_train == l) , :));
end
clear u s
rank_accuracy = NaN(size(ranks2try));
for k=ranks2try
	tic
	predicted_labels_valid = NaN(size(y_valid));
	mins_valid = inf(size(y_valid));
	for l=distinct_labels
		Vlk = V(:, 1:k, l+1);
		X_valid_approx = (Vlk*Vlk'*X_valid')';
		d = sum((X_valid - X_valid_approx).^2, 2);
		update_here = find(d < mins_valid);
		mins_valid(update_here) = d(update_here);
		predicted_labels_valid(update_here) = l;
	end
	accuracy = sum(y_valid==predicted_labels_valid) / m_valid;
	rank_accuracy(find(ranks2try==k)) = accuracy;
	printf(['rank = ' num2str(k) '  |  Accuracy: ' num2str(accuracy) '  |  '])
	toc
end

plot(ranks2try, rank_accuracy)

printf('\nTesting Stage:\n')
[max_acc, max_acc_idx] = max(rank_accuracy);
best_k = ranks2try(max_acc_idx)
predicted_labels_test = NaN(size(y_test));
mins_test = inf(size(y_test));
for l=distinct_labels
	[Ul, Sl, Vl] = svd(X_train( find(y_train == l) , :));
	Vlk = Vl(:, 1:best_k);

	% try out this label's approximation for X_test 
	X_test_approx = (Vlk*Vlk'*X_test')';
	d = sum((X_test - X_test_approx).^2, 2);
	update_here = find(d < mins_test);
	mins_test(update_here) = d(update_here);
	predicted_labels_test(update_here) = l;
end

accuracy = sum(y_test == predicted_labels_test) / m_test
toc