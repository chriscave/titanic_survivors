function [lambda_vect, error_train, error_cv] =  validationCurve(X, y, X_cv, y_cv, theta, alpha, num_iters, rowct, colct);
  lambda_vect = [0 0.001 0.003 0.01 0.03 0.1 0.3 0.5 1 3 5 10]';
  error_train =[];
  error_cv = [];
  for i = 1:length(lambda_vect);
    lambda = lambda_vect(i);
    [theta,C_H] = gradientDescent(X,y,theta,lambda,alpha,num_iters,rowct,colct);
    error_train = [error_train ;Cost(X,y,theta,0,rowct,colct)];
    error_cv = [error_cv; Cost(X_cv,y_cv,theta,0,rowct,colct)];
  end
end