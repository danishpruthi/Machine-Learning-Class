function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    theta_temp = zeros(size(X, 2), 1);
    i = 1;
    while i <= size(X,2)
        theta_temp(i, 1) = theta(i, 1) - alpha*((1/m)*(dot(X*theta - y, X(:, i))));
        i = i + 1;
    end
    i = 1;
    while i <= size(X,2)
        theta(i, 1) = theta_temp(i, 1);
        i = i + 1;
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
