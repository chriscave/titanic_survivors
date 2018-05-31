function J = Cost(X,y,theta,lambda,rowct,colct);
  m = size(X,1);
  r = size(rowct,2);
  hyprowct = [1];
  for i = 2:r;
    k = hyprowct(i-1) + (rowct(i)-rowct(i-1)) + 1;
    hyprowct = [hyprowct,k];
  end
  H = hyp(X,theta,rowct,colct)(hyprowct(r-1),:);
  colct_= colct;
  colct_(r) = [];
  theta_ = theta;
  theta_(:,colct_) = [];
  reg = sumsq(theta_, dim=1);
  J = 1/m *( -y' * log(H') - ((1-y)' * log(1 - H')) + (lambda / 2) *sum(reg));
  
end