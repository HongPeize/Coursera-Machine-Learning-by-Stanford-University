function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    for i = 1: size(X,2)
        for j = 1 : m
            h = theta' * X(j,:)'; %prediction
            theta(i,:) = theta (i,:) - (alpha/m) * (h - y(j,:)) * X(j,i);
            %theta
        end
    end
    % ============================================================
    theta_history(iter,:) = theta;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    if iter > 1 %&& iter < 1500
        if J_history(iter) > J_history(iter-1)
            %iter
            [theta(1),theta(2)] = theta_history(iter-1,:);
            break
        end
    end
end
