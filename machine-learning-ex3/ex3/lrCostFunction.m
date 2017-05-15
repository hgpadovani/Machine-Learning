function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

pred = X*theta;
predictions = sigmoid(pred);% predictions of hypothesis on all m examples

%calculando para y=1
aux = log(predictions);
pos = (aux')*y;

%calculando para y=0
aux1 = log(1-predictions);
aux2 = 1-y;
neg = (aux1')*aux2;

total = -(pos+neg); 

reg = theta.^2;
a = (lambda)/(2*m);
reg = a*reg;
b = reg(2:length(theta),:);
reg = sum(b);

J = (1/m)*total+reg;

%%%%%%%%%%%%Calculando o Gradiente%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%Calculando o Gradiente%%%%%%%%%%%%%%%%%%%%

h = predictions;
error = h-y;
grad1 = X'*error;

grad1 = (1/m)*grad1;
aux4 = grad1(2:length(theta),:)+ (lambda/m)*theta(2:length(theta),:);
grad(1,1) = grad1(1,1);
grad(2:length(theta),:) = aux4;
% =============================================================

grad = grad(:);

end
