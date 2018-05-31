function H = hyp(X, theta, rowct, colct);
  r = size(rowct,2);
  H = [];
  activation = X';
  for i = 1: (r - 2);
    weights = theta((rowct(i):rowct(i+1) - 1), (colct(i):colct(i+1) - 1));
    activation = [ones(1,size(X,1));sigmoid(weights * activation)];
    H = [H;activation];
  end
  weights = theta((rowct(r - 1): rowct(r) - 1), ( colct(r - 1): colct(r) - 1));
  H = [H;sigmoid(weights * activation)];
  