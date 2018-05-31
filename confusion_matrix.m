function CM = confusion_matrix(X_cv,y_cv,threshold,theta,rowct,colct);
  CM = zeros(2,2);
  H = hyp(X_cv, theta, rowct,colct);
  y_cv_prob = H(end,:);
  surv_index = find(y_cv);
  A = y_cv_prob(surv_index);
  k = find(y_cv_prob(surv_index) > threshold);
  A(k) = 1;
  k = find (y_cv_prob(surv_index) <= threshold);
  A(k) = 0;
  CM(1,1) = sum(A);
  CM(1,2) = length(A) - sum(A);
    
  dead_index = find(~y_cv);
  B = y_cv_prob(dead_index);
  k = find(y_cv_prob(dead_index) > threshold);
  B(k) = 1;
  k = find (y_cv_prob(dead_index) <= threshold);
  B(k) = 0;
  CM(2,1) = sum(B);
  CM(2,2) = length(B) - sum(B);
  
  
end