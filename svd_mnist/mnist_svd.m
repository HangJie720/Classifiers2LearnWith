close all
clear all

ranks2try = [3];

load ../datasets/mnist/MNIST.mat;  % X_train, y_train, X_test, y_test
X_train = double(X_train);
X_test = double(X_test);
distinct_labels = unique([y_train', y_test']);

% reshape images to 1D
[m_train, w, h] = size(X_train);
X_train = reshape(X_train, [m_train, w*h]);
[m_test, w, h] = size(X_test);
X_test = reshape(X_test, [m_test, w*h]);

% Let's shuffle, just in case
train_shuffle = randperm(m_train);
X_train = X_train(train_shuffle, :);
y_train = y_train(train_shuffle, :);
test_shuffle = randperm(m_test);
X_test = X_test(test_shuffle, :);
y_test = y_test(test_shuffle, :);

% Let's take 10k of the training examples to serve as a
% validation set
m_valid = 10^4;
X_valid = X_train(1:m_valid, :);
y_valid = y_train(1:m_valid, :);
m_train = m_train - m_valid;
X_train = X_train(m_valid + 1: end, :);
y_train = y_train(m_valid + 1: end, :);

% normalize data
% X = X - mean(X);  % hurts accuracy
% X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

printf('\nCheck Balance:\n')
% printf('Totals -- ')
% for l=distinct_labels
% 	printf([num2str(sum(y==l)) ' '])
% end
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