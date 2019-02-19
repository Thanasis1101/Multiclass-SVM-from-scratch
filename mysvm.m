% ===== Inputs ===== %

train_set_inputs = readMNISTImages('train-images.idx3-ubyte')';
train_labels = readMNISTLabels('train-labels.idx1-ubyte');
test_set_inputs = readMNISTImages('t10k-images.idx3-ubyte')';
test_labels = readMNISTLabels('t10k-labels.idx1-ubyte');

C = 2;
number_of_samples_from_every_class = 500; % set to Inf for all samples
kernel_name = 'linear';
param = 1;
epsilon = 10^-4; % from 0 to epsilon all alphas are cosidered as zero

% ===== Initializations ===== %

% Sort inputs and labels according to labels
[train_labels,index]=sort(train_labels);
train_set_inputs=train_set_inputs(index, :);

% calculate the index of the first occurence for every label
all_possible_labels = unique(train_labels);
number_of_unique_labels = size(all_possible_labels, 1);
starting_indices = zeros(number_of_unique_labels + 1, 1);
for test_index=all_possible_labels'
    starting_indices(test_index+1) = find(train_labels==test_index,1);
end
starting_indices(end) = size(train_labels, 1);

% seperate the classes for the SVM
classes = cell(number_of_unique_labels, 1);
for test_index=1:number_of_unique_labels
    % classes(i) is a matrix with a number of samples from the same class
    % (number_of_samples_from_every_class x dimensions)
    from = starting_indices(test_index);    
    next_index = starting_indices(test_index+1);
    % if this class has less samples than number_of_samples_from_every_class
    % then just take all the samples of the class
    to = min(starting_indices(test_index)+number_of_samples_from_every_class-1, next_index);
    classes(test_index) = {train_set_inputs(from:to, :)};
end


% ===== Training ===== %

% solve all possible different SVMs (1 vs 1 approach)
number_of_svms = number_of_unique_labels*(number_of_unique_labels-1)/2;
svms = cell(number_of_svms, 5);

start_time = cputime;

svm_counter = 0;
for i=1:number_of_unique_labels       
    for j=i+1:number_of_unique_labels
        
        svm_counter = svm_counter + 1;
        
        % SVM for i (y=1) vs j (y=-1) class
        
        x = [
            cell2mat(classes(i));
            cell2mat(classes(j))
            ];
        
        y = [
            ones(size(cell2mat(classes(i)), 1), 1);
            -ones(size(cell2mat(classes(j)), 1), 1)
            ];
        
        % solve and keep the results

        [alpha, support_vectors_x, support_vectors_y, bias] = fit_training_data(x, y, C, epsilon, kernel_name, param);
        svms(svm_counter, 1) = {alpha};
        svms(svm_counter, 2) = {support_vectors_x};
        svms(svm_counter, 3) = {support_vectors_y};
        svms(svm_counter, 4) = {bias};
        
        % calculate the percision acording to training set
        % (for use as confidence on voting while testing)

        corrects=0;
        number_of_tests = size(x,1);
        for test_index=1:number_of_tests
            inp = x(test_index, :);
            
            % calculate result = w*Ö(inp) which is sum(part_of_w*K(x, inp))
            % where part_of_w has been calculated as a*y
            result = sum(alpha.*support_vectors_y.*kernel(kernel_name, support_vectors_x, inp, param))+bias;
            
            correct_sign = y(test_index);
            if sign(result) == correct_sign
                corrects = corrects + 1;
            end

        end
        
        precision = corrects/number_of_tests;        
        svms(svm_counter, 5) = {precision};
        
    end
    
    
end

training_time = cputime - start_time;

% ===== Testing ===== %

corrects=0;
number_of_tests = size(test_set_inputs, 1);

start_time = cputime;

for test_index=1:number_of_tests
    inp = test_set_inputs(test_index, :);
    correct_label = test_labels(test_index);
    votes = zeros(number_of_unique_labels, 1);    
    svm_counter = 0;
    
    % test for the i vs j svm
    
    for i=1:number_of_unique_labels
        for j=i+1:number_of_unique_labels
            svm_counter = svm_counter + 1;
            alpha = cell2mat(svms(svm_counter, 1));
            support_vectors_x = cell2mat(svms(svm_counter, 2));
            support_vectors_y = cell2mat(svms(svm_counter, 3));
            bias = cell2mat(svms(svm_counter, 4));
            precision = cell2mat(svms(svm_counter, 5));
            
            % calculate result = w*Ö(inp) which is sum(part_of_w*K(x, inp))
            % where part_of_w has been calculated as a*y            
            result = sum(alpha.*support_vectors_y.*kernel(kernel_name, support_vectors_x, inp, param))+bias;
            
            if sign(result) == 1 % class i won
                votes(i) = votes(i) + precision;
            else % class j won
                votes(j) = votes(j) + precision;                
            end
        end
    end
    [~, max_result_index] = max(votes);
    predicted_label = max_result_index - 1;    
    
    if predicted_label == correct_label
        corrects = corrects + 1;
    end
    
end

testing_time = cputime - start_time;
precision = corrects/number_of_tests;
fprintf(1,'Precision: %g / %g (%g%%)\n', corrects, number_of_tests, 100*precision);
fprintf(1,'Seconds for training: %g\n', training_time);
fprintf(1,'Seconds for testing: %g\n', testing_time);

