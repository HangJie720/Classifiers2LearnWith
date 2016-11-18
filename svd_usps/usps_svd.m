close all
clear all

% Results with no pre-processing of included dataset:
% best_k (rank) =  11 
% accuracy =  0.94513
% see 'results.pdf'
% ranks2try = [1:256];  % this will take a while, maybe just check 5:20
ranks2try = 3

load ../datasets/usps/USPS.mat;
X = fea;
y = gnd;
[m, n] = size(X);
distinct_labels = unique(y)';

% normalize data
% X = X - mean(X);  % hurts accuracy
% X = X./mean(sum((X-mean(X)).^2));  % doesn't help accuracy

m_train = round(0.6*m);
X_train = X(1:m_train, :);
y_train = y(1:m_train);

m_valid = round(0.2*m);
X_valid = X(m_train + 1 : m_train + m_valid, :);
y_valid = y(m_train + 1 : m_train + m_valid);

m_test = m - m_valid - m_train;
X_test = X(m_train + m_valid + 1 : end, :);
y_test = y(m_train + m_valid + 1 : end);


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

% tic
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
% toc