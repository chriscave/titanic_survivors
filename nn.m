function [prob,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X,y);
 [theta, Cost_history] = gradientDescent(X, y, theta1, theta2, lambda, alpha, num_iters);
 J = Cost_history(num_iters);
 [m1 n1] = size(theta1);
 [m2 n2] = size(theta2);
 theta1 = theta([1:m1],[1:n1]);
 theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
 [p,prob] = predict(theta1, theta2, X,y);
end