function J = Cost(theta1, theta2, theta3, X,y,lambda);
  J = 0;
  [a2, a3, a4] = hyp(X,theta1,theta2,theta3);
  [m n] = size(X);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  [m3 n3] = size(theta3);
  theta1_reg = theta1(:, [2:n1]);
  theta2_reg = theta2(:, [2:n2]);
  theta3_reg = theta3(:, [2:n3]);
  lambda_reg = (lambda/2) *( sum(sumsq(theta1_reg)) + sum(sumsq(theta2_reg)) + sum(sumsq(theta3_reg)));
  J = 1/m * ((-y' * log(a4')) - ((1 - y)' * log( 1 - a4')) +lambda_reg);
end