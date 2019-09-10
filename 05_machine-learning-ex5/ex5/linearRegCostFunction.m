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

h = X * theta;

theta_rest = theta(2:end);

% regularized paramter
r = lambda / 2 / m * sum(theta_rest .^ 2);

J = 1 / 2 / m * sum((h - y) .^ 2) + r;

% for j = 0
X_0 = X(:,1);
grad_0 = 1 / m * [X_0' * (h - y)];

% for j >= 1
X_rest = X(:,2:end);
grad_rest = 1 / m * [X_rest' * (h - y)] + lambda / m * theta_rest;

grad = [grad_0; grad_rest];
% =========================================================================

grad = grad(:);

end
