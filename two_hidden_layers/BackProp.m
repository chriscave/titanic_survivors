function [D1,D2,D3] = BackProp(X,y,theta1,theta2,theta3,lambda);
  [m n] = size(X);  
  
  [a2, a3, a4] = hyp(X, theta1, theta2, theta3); #forward propagation
  
  delta4 = a4 - y';
  
  delta3 = (theta3' * delta4).*a3.*(1-a3);
  #when finding the derivatives, one does not find the derivative of theta_{0,j} because it does not exist, so we delete the first row.
  delta3([1],:) = []; 
  
  delta2 = (theta2' * delta3).*a2.*(1-a2);
  delta2([1],:) = []; #when finding the derivatives, one does not find the derivative of theta_{0,j} because it does not exist, so we delete the first row.
    
  Delta1 = delta2 * X;
  Delta2 = delta3 * a2';
  Delta3 = delta4 * a3';
  
  [m1 n1] = size(Delta1);
  [m2 n2] = size(Delta2);
  [m3 n3] = size(Delta3);
  
  D1 = zeros(m1, n1);
  D2 = zeros(m2, n2);
  D3 = zeros(m3, n3);
  
  
  E1 = 1/m * ( Delta1(:,[2:n1]) + lambda * theta1(:,[2:n1]));
  e1 = 1/m * ( Delta1(:,[1]));
  
  E2 = 1/m * ( Delta2(:,[2:n2]) + lambda * theta2(:,[2:n2]));
  e2 = 1/m * ( Delta2(:,[1]));
  
  E3 = 1/m * ( Delta3(:,[2:n3]) + lambda * theta3(:,[2:n3]));
  e3 = 1/m * ( Delta3(:,[1]));
  
  
  D1 = [e1, E1];
  D2 = [e2, E2];
  D3 = [e2, E3];
end