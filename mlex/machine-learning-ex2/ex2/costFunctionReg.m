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

[noRegJ, noRegGrad] = costFunction(theta, X, y); % use previous costFunction

reg_theta = sum(theta.^2) - theta(1)^2; % without θo
J = noRegJ + (lambda / (2 * m)) * reg_theta;

grad(1) = noRegGrad(1);
grad(2:size(theta)) = noRegGrad(2:size(theta)) + (lambda / m) * theta(2:size(theta));

% =============================================================

end
