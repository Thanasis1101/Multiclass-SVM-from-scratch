function result = kernel(kernel_name, x1, x2, param)
%  perfors the K(x1, x2) kernel and returns the results
%
%  Parameters: kernel_name - string with the name of the kernel function
%              x1 - the whole matrix x1
%              x2 - the whole matrix x2
%              param - the value of the parameter of the kernel function
%
%  Values for kernel_name:
%              'linear'     (default)
%              'polynomial' - param is the d (dimension)
%              'rbf'        - param is the g (gama = 1/(2*s^2))

    switch kernel_name
        case 'linear'
            result = x1*x2';
        case 'polynomial'
            d = param;
            result = (x1*x2' + 1).^d;            
        case 'rbf'
            R1 = size(x1,1); % R1 = number of rows of x1
            R2 = size(x2,1); % R2 = number of rows of x2
            result = ones(R1, R2);
            for i=1:R1
                for j=1:R2
                    g = param;
                    diff = x1(i,:)-x2(j,:);
                    result(i,j) = exp(-g*(diff*diff'));
                end
            end
        otherwise
            result = x*x';
    end
    

end

