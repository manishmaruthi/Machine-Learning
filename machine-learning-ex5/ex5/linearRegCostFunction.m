function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters

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

J = ((1./(2*m))*sum((h-y).^2)) + (lambda./(2*m))*(sum(theta.^2)-theta(1)^2);

temp1 = ((h - y) .* X(:,1));
grad(1) = ((1./m).* sum(temp1));
if 0
	temp2 = ((h - y) .* X(:,2)) + ((lambda./m).* theta(2));
	grad(2) = ((1./m).* sum(temp2));
end
if 1
	for i = 2 : size(theta)
		temp2 = ((h - y) .* X(:,i)) + ((lambda./m).* theta(i));
		grad(i) = ((1./m).* sum(temp2));
	end
end




% =========================================================================

grad = grad(:);

end
