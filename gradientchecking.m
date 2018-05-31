function D_app = gradientchecking(X,y,theta,lambda,epsilon,rowct,colct)
  [m,n] = size(theta);
  ep = zeros(m,n);
  D_app = [];
  for i = 1:m;
    for j = 1:n;
      ep(i,j) = epsilon;
      app = (1 / (2 * epsilon)) * (Cost(X,y, (theta + ep), lambda, rowct,colct) - Cost(X,y, (theta - ep), lambda, rowct,colct));
      D_app = [D_app; app];
      ep = zeros(m,n);
    end
  end
D_app = reshape(D_app,n,m)';
end
  