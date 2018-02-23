function [theta, Cost_history] = gradientDescent(X, y, theta1, theta2, lambda, alpha, num_iters);
  Cost_history = zeros(num_iters, 1);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  for iter = 1:num_iters;
    theta = blkdiag(theta1, theta2);
    [D1, D2] = BackProp(X,y,theta1, theta2, lambda);
    theta = theta - alpha * blkdiag(D1,D2);
    theta1 = theta([1:m1],[1:n1]);
    theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
    Cost_history(iter) = Cost(theta1, theta2, X,y,lambda);
  end
end
