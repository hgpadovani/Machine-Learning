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

%%%foward propagation

tmp_eye = eye(num_labels);
y_matrix = tmp_eye(y,:);

a1 = [ones(m, 1) X]; %size = 5000x401
z2 = a1*Theta1'; %size = 5000x25
a2 = sigmoid(z2);
a2 = [ones(m,1) a2]; %size = 5000x26
z3 = a2*Theta2'; %size = 5000x10
a3 = sigmoid(z3);

%%%%%%%%%%%%Calculando o Custo%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calculando para y=1
aux = log(a3);
pos = aux.*y_matrix;

%calculando para y=0
aux1 = log(1-a3);
aux2 = 1-y_matrix;
neg = aux1.*aux2;

total = -(pos+neg); 
total = sum(sum(total));

J = (1/m)*total;

%%%%%%%%%%%%%%%%%%Calculando a Regulariza??o%%%%%%%%%%%%%%%%%

reg1 = (Theta1).^2;
tam1 = size(Theta1);
b1 = reg1(:,2:tam1(2)); %taking the first column

reg2 = (Theta2).^2;
tam2 = size(Theta2);
b2 = reg2(:,2:tam2(2)); %taking the first column

b1 = sum(sum(b1));
b2 = sum(sum(b2));
reg_sum = b1 + b2;
a = (lambda)/(2*m);
reg = a*reg_sum;


J = J+reg;

%%%%%%%%%%%%%%%% Backpropagation Algorithm %%%%%%%%%%%%%%%%%

d3 = a3 - y_matrix; %size 5000x10
d2 = d3*Theta2(:,2:length(Theta2));%size 5000x25
sig = sigmoidGradient(z2); %size 5000x25
d2 = sig.*d2; %size: 5000x25

delta2 = d3'*a2; %size 10x26
delta1 = d2'*a1; %size 25x401

Theta1_grad = delta1*(1/m);
Theta2_grad = delta2*(1/m);

%%%%%%%%%%%%% Calculando a Regulariza??o do Backpropagation %%%%%%%%%%%%%

reg_theta1 = Theta1(:,2:tam1(2));
reg_theta2 = Theta2(:,2:tam2(2));

reg_theta1 = (lambda/m)*reg_theta1;
reg_theta2 = (lambda/m)*reg_theta2;
reg_theta1 = [zeros(tam1(1), 1) reg_theta1]; 
reg_theta2 = [zeros(tam2(1), 1) reg_theta2]; 

Theta1_grad = Theta1_grad+reg_theta1;
Theta2_grad = Theta2_grad+reg_theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
