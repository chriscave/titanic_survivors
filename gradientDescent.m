function [theta, Cost_history] = gradientDescent(X,y,theta,lambda,alpha,num_iters,rowct,colct);
  Cost_history = [];
  r =size(rowct,2);
  
  for iter = 1:num_iters;    
    D = BackProp(X,y,theta,lambda,rowct,colct);    
    theta = theta - alpha * D;
    Cost_history = [Cost_history;Cost(X,y,theta,lambda,rowct,colct)'];
  end
end