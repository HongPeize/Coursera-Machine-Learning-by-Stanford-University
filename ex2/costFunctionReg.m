function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
a = 0;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost function
for i = 1 : m
    z = theta' * X(i,:)';
    h = sigmoid(z);
    J = J + (1/m) * (-y(i,:) * log(h) - (1-y(i,:)) * log(1-h));
end

for j = 2 : size(X,2)
    a = a + (lambda/(2*m)) * theta(j,:)^2;
end

J = J + a;
% =============================================================

%j=0
for i = 1 : m
    z = theta' * X(i, :)';
    h = sigmoid(z); %prediction
    grad(1,:) = grad(1,:) + (1/m) * (h - y(i,:)) * X(i,1);
end
    
    
%j>=1

for i = 2 : size(X,2)
    
    for j = 1 : m
        z = theta' * X(j, :)';
        h = sigmoid(z); %prediction
        grad(i,:) = grad(i,:) + (1/m) * (h - y(j,:)) * X(j,i);
    end
    
    grad(i,:) = grad(i,:) + (lambda/(m)) * theta(i,:)^2;
    
end
    
end
