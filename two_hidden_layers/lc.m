function err_prob = lc(X,y,theta1, theta2, theta3, num_iters,alpha,lambda);
  [m,n] = size(X);
  c = floor(m / 20);
  domain = c + 1
  err_prob = zeros(domain, 3);
  k = randperm(m);  
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  [m3 n3] = size(theta3);
  
  for i = 1:c;
    err_prob(i,1) = i;
    X_train = X(k(1: (12 * i)),:);
    X_cv = X(k((12 * i + 1) : 16 * i),: );
    X_test = X(k((16 * i + 1) : 20 * i, :));
    y_train = y(k(1: (12 * i)),:);
    y_cv = y(k((12 * i + 1) : 16 * i));
    y_test = y(k((16 * i + 1) : 20 * i));
    
    [prob_train,theta, Cost_history, J] = nn(theta1, theta2, theta3, alpha, lambda, num_iters, X_train,y_train);
    err_prob(i,2) = 1 - prob_train
    theta1 = theta([1:m1],[1:n1]);
    theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
    theta3 = theta([(m1 + m2 + 1): m1 + m2 + m3], [(n1 + n2 +1): (n1 + n2 + n3)]);
    
    [p, prob_test]= predict(theta1, theta2, theta3, X_cv,y_cv);
    err_prob(i,3) = 1 - prob_test;
  end
end   
  