function [p,prob] = predict(theta1, theta2, theta3, X,y);
 m = size(X, 1);
 prob = 0;
 
 p = zeros(size(X, 1), 1);
 [a2,a3,a4] = hyp(X,theta1,theta2,theta3);
 
 p = round(a4);
 dif = abs(p-y');
 sum_dif = sum(dif(:) == 1);
 prob = 1 - (sum_dif / length(y));
end 