function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
[m, n] = size(z);
i = 1; j = 1;
while i <= m
    j = 1;
    while j <= n
        g(i, j) = 1.0/(1 + exp(-1*z(i, j)));
        j = j + 1;
    end
    i = i + 1;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
