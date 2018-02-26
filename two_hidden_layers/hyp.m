function [a2,a3,a4] = hyp(X,theta1, theta2, theta3);
  [x1 x2] = size(X);
  m1 = size(theta1)(1);
  m2 = size(theta2)(1);
  m3 = size(theta3)(1);
  
  a2 = zeros(m1+1,x1);
  a3 = zeros(m2+1,x1);
  a4 = zeros(m3,x1);
  
  
  z2 = theta1 * X';
  b2 = sigmoid(z2);
  n = size(b2)(2);
  a2 = [ones(1,n);b2];
  
  z3 = theta2 * a2;
  b3 = sigmoid(z3); #output is a row vector
  n = size(b3)(2);
  a3 = [ones(1,n);b3];
  
  z4 = theta3 * a3;
  b4 = sigmoid(z4); #output is a row vector
  a4 = b4;
end