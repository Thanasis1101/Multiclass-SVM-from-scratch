function [alpha, support_vectors_x, support_vectors_y, bias] = fit_training_data(x, y, C, epsilon, kernel_name, param)
%  Performs the qp optimisation and returns alpha (solutions of QP), the
%  support vectors (the samples from x that are the supported vectors),
%  the corresponding signs of supported vectors (y) and the bias
%
%  Parameters: x - the whole matrix x with the inputs
%              y - the whole matrix y with the corresponding class (-1, 1)
%              C - soft margin parameter
%              epsilon - 0 to epsilon (e.g. 10^-4) are considered as zeros
%              kernel_name - string with the name of the kernel function
%              param - the value of the parameter of the kernel function
%
%  Values for kernel_name:
%              'linear'     (default)
%              'polynomial' - param is the d (dimension)
%              'rbf'        - param is the g (gama = 1/(2*s^2))


    R = size(x, 1); % R = number of input samples
    
    if strcmp(kernel_name, 'linear')
        param = 0; % dump number in order to call the kernel function below
    end
    
    % ===== Solve QP ===== %

    % quadprog syntax: 1/2*x'*H*x + C*f'*x
    % minimize: 1/2*sum((a*(y*y').*(Ö(x)*Ö(x'))*a') - sum(a)
    % quadprog for current problem: x=a, H=(y*y').*(Ö(x)*Ö(x')), f=-ones
    % subject to: 0 <= a <= C, sum(a.*y) = 0
    
    % objective function: 1/2*sum((a*(y*y').*(Ö(x)*Ö(x'))*a') - sum(a)
    f = -ones(R, 1); 
    H = (y*y').*kernel(kernel_name, x, x, param);    
    % constraint: sum(a.*y) = 0
    Aeq = y';
    beq = 0;
    % constraint: 0 <= a <= C
    lb = zeros(R, 1);
    ub = C*ones(R,1);
    
    % Solve QP without showing help message
    options =  optimset('Display','off');
    alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
    
    % ===== Calculate return values ===== %
        
    support_vectors_indices = find(alpha > epsilon & alpha < (C - epsilon));
    
    % seperate and keep only support vectors    
    alpha = alpha(support_vectors_indices, :);
    support_vectors_x = x(support_vectors_indices, :);    
    support_vectors_y = y(support_vectors_indices, :);
    
    % calculate bias
    bias = (1/size(support_vectors_indices, 1))*sum(support_vectors_y - H(support_vectors_indices,support_vectors_indices)*alpha.*support_vectors_y);

    
end

