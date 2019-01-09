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
error = h - y;
sqr_error = error.^2;
sum_sqr_error = sum(sqr_error);
J_unreg = sum_sqr_error / (2 * m);
theta(1)=0;
theta_sqr = theta' * theta;
J_reg = sum(theta_sqr)*(lambda/(2*m));
J = J_unreg + J_reg;

grad_unreg = (X'*error/m);
grad_reg = ((lambda/m)*theta);
grad = grad_unreg + grad_reg;

% =========================================================================

grad = grad(:);

end
