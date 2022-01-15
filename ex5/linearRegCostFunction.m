function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%cost function
for i = 1:m
    h = theta' * X(i,:)';
    J = J + (1/(2*m)) * (h - y(i,:))^2;
end

for j = 2:size(theta)
    J = J + lambda/(2*m) * theta(j)^2;
end

%gradient decent
grad(1) = 1/m * sum( ((theta' * X')' - y) .* X(1:end,1) );

for i = 2:size(theta)
    h = (theta' * X')';
    grad(i) = 1/m * sum( (h - y) .* X(1:end,i), "all" ) + lambda/m * theta(i);
end

% =========================================================================

grad = grad(:);

end
