function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
[m n]= size(X); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

       t=zeros(n,1);   
    for j=1:n
       G=zeros(m,1);
       for i=1:m
         G(i) = G(i) + (X(i,:)*theta - y(i))*X(i,j);
       end
       t(j) = sum(G) / m ;      
     
    
    end
    theta = theta - alpha * t ;  
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
