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
K = num_labels;
L = hidden_layer_size;
N = input_layer_size;
         
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

%Part 1
X = [ones(m,1) X];

z2= X*Theta1';

a = sigmoid(z2);

a = [ones(m,1) a];

z3 = a * Theta2';

h = sigmoid(z3);

for i=1:m
	temp(i)=0;
	for k=1:K
		y1(i,k)=0;
	end
end

for i=1:m
	for j=1:K
		if(y(i)==j)
			y1(i,j)=1;
		end
	end	
end

for i=1:m
	for k=1:K
		temp(i)+= (-y1(i,k).*log(h(i,k))) - ((1-y1(i,k)).*(log(1-h(i,k)))); 
	end
end

%Without Regularization
if 0
J = ((1/m).* sum(temp));
end

%With Regularization
if 1
for j=1:L
	temp2(j)=0;
end

for j=1:L
	for k=2:(N+1) 
		temp2(j)+= Theta1(j,k).^2;
	end
end

for j=1:K
	temp3(j)=0;
end

for j=1:K
	for k=2:(L+1) 
		temp3(j)+= Theta2(j,k).^2;
	end
end

J = ((1/m).* sum(temp)) + ((lambda./(2*m)).* (sum(temp2) + sum(temp3)));

%Backpropagation Algorithm to find Gradient

delta_3= h-y1;
	
z2_temp= [ones(m,1) z2];
	
delta_2= ((Theta2)' * (delta_3)') .* (sigmoidGradient(z2_temp))';
	
delta_2= delta_2( 2 : end, :);
	
Big_delta_1 = delta_2 * X;
	
Big_delta_2 = delta_3' * a;
	
if 0 % Without Regularization
	Theta1_grad = Big_delta_1./m;
	
	Theta2_grad = Big_delta_2./m;
end
	
for j=1:L
	for k=1:(N+1)
		if (k==1)
			Theta1_grad(j,k) = Big_delta_1(j,k)./m;
		else
			Theta1_grad(j,k) = (Big_delta_1(j,k)./m) + (lambda.* Theta1(j,k)./m) ;
		end
	end
end

for j=1:K
	for k=1:(L+1)
		if (k==1)
			Theta2_grad(j,k) = Big_delta_2(j,k)./m;
		else
			Theta2_grad(j,k) = (Big_delta_2(j,k)./m) + (lambda.* Theta2(j,k)./m) ;
		end
	end
end
	

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
