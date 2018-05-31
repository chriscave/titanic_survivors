function [error_train, error_cv] = learningCurve(X, y, X_cv, y_cv, lambda, theta, alpha, num_iters, rowct, colct)
  error_train = [];
  error_cv = [];
  j = floor(size(X,1) / 50);
  for i = 1:50;
    [theta, Cost_history] = gradientDescent(X(1:(i*j),:),y(1:(i*j),:),theta,lambda,alpha,num_iters,rowct,colct);
    error_train = [error_train; Cost(X(1:(i*j),:),y(1:(i*j),:),theta,lambda,rowct,colct)];
    error_cv = [error_cv; Cost(X_cv,y_cv,theta, lambda, rowct, colct)];
  end
end