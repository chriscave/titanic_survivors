function [D1 D2] = BackProp(X,y,theta1,theta2,lambda);
  [m n] = size(X);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  
  D1 = zeros(m1, n1);
  D2 = zeros(m2, n2);
  Delta1 = zeros(m1, n1);
  Delta2 = zeros(m2, n2);
  
  [a2 a3] = hyp(X, theta1, theta2);
  delta3 = a3 - y;
  delta2 = (theta2' * delta3).*a2.*(1-a2);
  
  Delta1 = Delta1 + delta2 * a1';
  Delta2 = Delta2 + delta3 * a2';
  
  E1 = 1/m * ( Delta1(:,[2,n1]) + lambda theta1(:,[2:n1]));
  e1 = 1/m * ( Delta1(:,[1]));
  
  E2 = 1/m * ( Delta2(:,[2,n2]) + lambda theta2(:,[2:n2]));
  e2 = 1/m * ( Delta2(:,[1]));
  
  D1 = [e1 , E1];
  D2 = [e2, E2];
end
  