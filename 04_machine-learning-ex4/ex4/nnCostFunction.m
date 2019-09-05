function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% compute hypothesis
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

% create num_labels vector
labels = 1:num_labels;

for i = 1:m
  for k = 1:num_labels
     yi = (labels == y(i));
     J += -yi(k) * log(h(i,k)) - (1 - yi(k)) * log(1 - h(i,k));
  endfor
endfor

theta1_sum = 0;
Theta1_rest = Theta1(:,2:end);
for j = 1:hidden_layer_size
  for k = 1:input_layer_size
    theta1_sum += Theta1_rest(j,k) ^ 2;
  endfor
endfor

theta2_sum = 0;
Theta2_rest = Theta2(:,2:end);
for j = 1:num_labels
  for k = 1:hidden_layer_size
    theta2_sum += Theta2_rest(j,k) ^ 2;
  endfor
endfor

% compute regularization parameter
r = lambda / 2 / m * (theta1_sum + theta2_sum);

% compute J
J = 1 / m * J + r;

% -------------------------------------------------------------

Delta_2 = zeros(size(Theta2));
Delta_1 = zeros(size(Theta1));

for t = 1:m
  %Step 1:Compute a3 for t
  a_1 = X(t,:)'; % 400*1
  a_1 = [1 ; a_1]; % 401*1
  z_2 = Theta1 * a_1; % 25*1 
  a_2 = sigmoid(z_2); % 25*1
  a_2 = [1 ; a_2]; % 26*1
  z_3 = Theta2 * a_2; % 10*1
  a_3 = sigmoid(z_3); % 10_1

  % Step 2: Compute delta in layer 3
  y_bin = (labels == y(t));
  delta_3 = a_3 - y_bin'; % 10*1
  
  % Step 3: Compute delta in layer 2
  delta_2 = Theta2' * delta_3 .* a_2 .* (1 - a_2); % 26*1
  %delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(z_2); %25*1
  
  % Step 4: Compute Delta each layer
  Delta_2 = Delta_2 + delta_3 * a_2';
  Delta_1 = Delta_1 + delta_2(2:end) * a_1';
  
endfor

% Step 5: Compute gradients

% j = 0
Theta1_grad_0 = 1 / m * Delta_1(:,1);
Theta2_grad_0 = 1 / m * Delta_2(:,1);

% j = 1
Theta1_grad_rest = 1 / m * Delta_1(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad_rest = 1 / m * Delta_2(:,2:end) + lambda / m * Theta2(:,2:end);
%Theta2_grad = 1 / m * Delta_2;

Theta1_grad = [Theta1_grad_0 Theta1_grad_rest];
Theta2_grad = [Theta2_grad_0 Theta2_grad_rest];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
