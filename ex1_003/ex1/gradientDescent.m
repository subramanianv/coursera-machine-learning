function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

[m n]= size(X);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    t=zeros(n,1);   
    for j=1:n
       G=zeros(m,1);
       for i=1:m
         G(i) = G(i) + (X(i,:)*theta - y(i))*X(i,j);
       end
       t(j) = sum(G) / m ;      
     
    % Save the cost J in every iteration    
    
    end
    theta = theta - alpha * t ;  
end

end
