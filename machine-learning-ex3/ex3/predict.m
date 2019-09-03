function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 以下の作業を繰り返す
% バイアス追加
% 重み付け
% シグモイド関数

% まず0番目に1（バイアス）を追加する
a_1 = [ones(m, 1) X];

% 重み付けをする
z_2 = a_1 * Theta1';

% シグモイド関数（g関数）に渡す
a_2 = sigmoid(z_2);

% 0番目に1（バイアス）を追加する
a_2 = [ones(m, 1) a_2];

% 重み付けをする
z_3 = a_2 * Theta2';

% シグモイド関数（g関数）に渡す
a_3 = sigmoid(z_3);

% 最大値のインデックスを取得する
[vals, idxs] = max(a_3,[],2);

p = idxs;
% =========================================================================


end
