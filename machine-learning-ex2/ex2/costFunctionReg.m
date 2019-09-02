function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

% regularized_parameter
r = lambda / 2 /m * theta(2,:);

J = -1 / m * [y' * log(h) + (ones(size(y)) -y)' * log(ones(size(h)) -h)] + r;

% for j = 0
X_0 = X(:,1);
grad_0 = 1 / m * [X_0' * (h - y)];

% for j >= 1
n = size(X)(2);
X_rest = X(:,2:n);
theta_rest = theta(2:length(theta));
grad_rest = 1 / m * [X_rest' * (h - y)] + lambda / m * theta_rest;

grad = [grad_0; grad_rest];

% =============================================================

end
