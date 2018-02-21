function [D1,D2] = BackProp(X,y,theta1,theta2,lambda);
  [m n] = size(X);
  [r1 r2] = size(theta2)
  
  
  
  [a2, a3] = hyp(X, theta1, theta2);
  [s1 s2] = size(a2);
  delta3 = a3 - y';
  theta2 = theta2(:,[2:r2]);
  a2 = a2([2:s2],:]);
  delta2 = (theta2' * delta3).*a2.*(1-a2);
  
  Delta1 = delta2 * X;
  Delta2 = delta3 * a2';
  [m1 n1] = size(Delta1);
  [m2 n2] = size(Delta2);
  
  D1 = zeros(m1, n1);
  D2 = zeros(m2, n2);
  
  
  E1 = 1/m * ( Delta1(:,[2:n1]) + lambda * theta1(:,[2:n1]));
  e1 = 1/m * ( Delta1(:,[1]));
  
  E2 = 1/m * ( Delta2(:,[2:n2]) + lambda * theta2(:,[2:n2]));
  e2 = 1/m * ( Delta2(:,[1]));
  
  D1 = [e1 , E1];
  D2 = [e2, E2];
end
  