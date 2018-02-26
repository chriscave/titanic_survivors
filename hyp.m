function [a2,a3] = hyp(X,theta1, theta2);
  [x1 x2] = size(X);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  
  a2 = zeros(m1+1,x1);
  a3 = zeros(m2, x1);
  
  z2 = theta1 * X';
  b2 = sigmoid(z2);
  [m n] = size(b2);
  a2 = [ones(1,n);b2];
  
  z3 = theta2 * a2;
  b3 = sigmoid(z3); #output is a row vector
  a3 = b3;
end