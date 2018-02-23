function [CVprob, mn, s] = crossVal(X,y,theta1, theta2,alpha, lambda, num_iters);
  mn = 0;
  s = 0;
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  CVprob = zeros(4,1);
  m = length(y);
  fl = floor(m / 4);
  X1 = X([1:(3*fl)],:);
  y1 = y([1:(3*fl)],:);
  X1_test = X([(3*fl + 1):m],:);
  y1_test = y([(3*fl + 1):m],:);
  
  X2 = X([(fl + 1):m],:);
  y2 = y([(fl + 1):m],:);
  X2_test = X([1: fl],:);
  y2_test = y([1: fl],:);
  
  X3 = [X([1:fl],:);X([2*fl + 1:m],:)];
  y3 = [y([1:fl],:);y([2*fl + 1:m],:)];
  X3_test = X([fl + 1: 2*fl],:);
  y3_test = y([fl + 1: 2*fl],:);
  
  X4 = [X([1:2*fl],:);X([3*fl + 1:m],:)];
  y4 = [y([1:2*fl],:);y([3*fl + 1:m],:)];
  X4_test = X([2*fl + 1:3*fl],:);
  y4_test = y([2*fl + 1:3*fl],:);
  
  #training on the folds
  [prob1,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X1,y1);
  theta11 = theta([1:m1],[1:n1]);
  theta12 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);  
  [a,a1] = hyp(X1_test,theta11, theta12);
  p1 = round(a1);
  
  [prob2,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X2,y2);
  theta21 = theta([1:m1],[1:n1]);
  theta22 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);  
  [b,a2] = hyp(X2_test,theta21, theta22);
  p2 = round(a2);
  
  [prob3,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X3,y3);
  theta31 = theta([1:m1],[1:n1]);
  theta32 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);  
  [c,a3] = hyp(X3_test,theta31, theta32);
  p3 = round(a3);
  
  [prob4,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X4,y4);
  theta41 = theta([1:m1],[1:n1]);
  theta42 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);  
  [d,a4] = hyp(X4_test,theta41, theta42);
  p4 = round(a4);
  
  
  
  dif1 = abs(p1-y1_test');
  dif2 = abs(p2-y2_test');
  dif3 = abs(p3-y3_test');
  dif4 = abs(p4-y4_test');
  
  sum_dif1 = sum(dif1(:) == 1);
  sum_dif2 = sum(dif2(:) == 1);
  sum_dif3 = sum(dif3(:) == 1);
  sum_dif4 = sum(dif4(:) == 1);
  
  CVprob(1) = 1 - (sum_dif1 / length(y1_test));
  CVprob(2) = 1 - (sum_dif2 / length(y2_test));
  CVprob(3) = 1 - (sum_dif3 / length(y3_test));
  CVprob(4) = 1 - (sum_dif4 / length(y4_test));
  
  mn = mean(CVprob);
  s = std(CVprob);
end