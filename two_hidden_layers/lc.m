function J = lc(X,y,theta1, theta2, theta3, num_iters,alpha,lambda);
  [m,n] = size(X);
  c = floor(m / 20);
  domain = c + 1
  J = zeros(domain, 3);
  k = randperm(m);  
  
  for i = 1:c;
    J(i,1) = i;
    X_train = X(k(1: (12 * i)),:);
    X_cv = X(k((12 * i + 1) : 16 * i );
    X_test = X(k((16 * i + 1) : 20 * i);
    y_train = y(k(1: (12 * i)),:);
    y_cv = y(k((12 * i + 1) : 16 * i );
    y_test = y(k((16 * i + 1) : 20 * i);
    
  