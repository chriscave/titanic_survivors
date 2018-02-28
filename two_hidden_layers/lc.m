function err_prob = lc(X,y,theta1, theta2, theta3, num_iters,alpha,lambda);
  [m,n] = size(X);
  step = 10; # Needs to be divisible by 10.
  c = floor(m / step);
  domain = c + 1;
  err_prob = zeros(domain, 4);
  k = randperm(m);
  k = k';
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  [m3 n3] = size(theta3);
  
  err_prob(domain,1) = domain;
  X_train = X(k(1: (step *.6 * domain)),:);
  X_cv = X(k((step *.6 * domain + 1) : (step*.8 * domain)),: );
  X_test = X(k((step * .8 * domain + 1) : m), :);
  y_train = y(k(1: (step * .6 * domain)));
  y_cv = y(k((step * .6 * domain + 1) : (step*.8 * domain)));
  y_test = y(k((step * .8 * domain + 1) : m));
  
  [prob_train,theta, Cost_history, J] = nn(theta1, theta2, theta3, alpha, lambda, num_iters, X_train,y_train);
  err_prob(domain,2) = 1 - prob_train;
  
  Theta1 = theta([1:m1],[1:n1]);
  Theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
  Theta3 = theta([(m1 + m2 + 1): m1 + m2 + m3], [(n1 + n2 +1): (n1 + n2 + n3)]);
  [p, prob_cv]= predict(Theta1, Theta2, Theta3, X_cv,y_cv);
  [p, prob_test] = predict(Theta1, Theta2, Theta3, X_test,y_test);
  err_prob(domain,3) = 1 - prob_cv;  
  err_prob(domain,4) = 1 - prob_test;
  
  for i = 1:c;
    err_prob(i,1) = i;
    X_train = X(k(1: ((step*.6) * i)),:);
    X_cv = X(k(((step*.6) * i + 1) : ((step*.8) * i)),: );
    X_test = X(k(((step*.8) * i + 1) : ((step) * i)), :);
    y_train = y(k(1: ((step*.6) * i)),:);
    y_cv = y(k(((step*.6) * i + 1) : (step*.8) * i));
    y_test = y(k(((step*.8) * i + 1) : (step * i)));
    
    [prob_train,theta, Cost_history, J] = nn(theta1, theta2, theta3, alpha, lambda, num_iters, X_train,y_train);
    err_prob(i,2) = 1 - prob_train;
    Theta1 = theta([1:m1],[1:n1]);
    Theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
    Theta3 = theta([(m1 + m2 + 1): m1 + m2 + m3], [(n1 + n2 +1): (n1 + n2 + n3)]);
    
    [p, prob_cv]= predict(Theta1, Theta2, Theta3, X_cv,y_cv);
    [p, prob_test] = predict(Theta1, Theta2, Theta3, X_test,y_test);
    err_prob(i,3) = 1 - prob_cv;  
    err_prob(i,4) = 1 - prob_test;
  end
  
  
end   
  