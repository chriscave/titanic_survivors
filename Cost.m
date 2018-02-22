function J = Cost(theta1, theta2, X,y,lambda);
  J = 0;
  [a2, a3] = hyp(X,theta1,theta2);
  [m n] = size(X);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  theta1_reg = theta1(:, [2:n1]);
  theta2_reg = theta2(:, [2:n2]);
  lambda_reg = (lambda/2) *( sum(sumsq(theta1_reg)) + sum(sumsq(theta2_reg)));
  J = 1/m * ((-y' * log(a3')) - ((1 - y)' * log( 1 - a3')) +lambda_reg);
end