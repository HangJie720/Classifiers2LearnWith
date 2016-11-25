close all
clear all

ranks2try = [11];

% load data/notMNIST_small.mat;
% % load data/notMNIST_large.mat;

% % reshape
% X = permute(images, [3 1 2]);
% [m h w] = size(X);
% X = reshape(X, [m h*w]);
% y = labels_unique';

% remove duplicates (this does some unwanted sorting also)
% [X, iuniq, iorig] = unique(X, 'rows');
% y = y(iuniq);
% printf(['\nWarning: ' num2str(m - length(y)) ' duplicates removed.\n'])
% [m, n] = size(X);

rootdir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(rootdir, 'data', 'notMNIST', 'notMNIST_small_no_duplicates.mat'))  % X, y

[m h w] = size(X);
n = w*h;
X = reshape(X, m, n);

% Let's shuffle
shuffle = randperm(m);
X = X(shuffle, :);
y = y(shuffle, :);

% define some variables for convenience
distinct_labels = unique(y)';

% Let's take 10k of the training examples to serve as a
% validation set and another 10k for a testing set
m_valid = min([round(m/5), 10^4]);
m_test = min([round(m/5), 10^4]);
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
rank_accuracy = NaN(size(ranks2try));
for k=ranks2try
	predicted_labels_valid = NaN(size(y_valid));
	mins_valid = inf(size(y_valid));
	for l=distinct_labels
		[Ul, Sl, Vl] = svd(X_train( find(y_train == l) , :));
		Vlk = Vl(:, 1:k);
		X_valid_approx = (Vlk*Vlk'*X_valid')';
		d = sum((X_valid - X_valid_approx).^2, 2);
		update_here = find(d < mins_valid);
		mins_valid(update_here) = d(update_here);
		predicted_labels_valid(update_here) = l;
	end
	accuracy = sum(y_valid==predicted_labels_valid) / m_valid;
	rank_accuracy(find(ranks2try==k)) = accuracy;
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