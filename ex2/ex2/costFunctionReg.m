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
theta_size=length(theta (:, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h= sigmoid(X*theta);

J= sum((y'*log(h))+((1-y')*log(1-h)))/(-m);
J+=(lambda/(2*m))*(sum(theta.*theta)-(theta(1)*theta(1)));

err= h-y; 
grad (1,1)=(1/m)* sum(err.*X(:, 1));

indexes=2:theta_size;
for i=indexes,
   grad (i,1)=(1/m)* sum(err.*X(:, i))+((lambda*theta(i))/m);
end;

%grad (2:theta_size,1)=(1/m)* sum(err.*X(:, 2:theta_size))+((lambda*theta(2:theta_size))/m);





% =============================================================

end
