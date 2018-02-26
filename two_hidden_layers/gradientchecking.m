function [App1,App2, App3] = gradientchecking(theta1, theta2, theta3, X,y,lambda,epsilon);
  [m1 n1] = size(theta1);
  [m2 n2] = size(theta2);
  [m3 n3] = size(theta3);
  
  App1 = zeros(m1,n1);
  App2 = zeros(m2,n2);
  App3 = zeros(m3,n3);
  for i = [1:m1];
    for j =[1:n1];
      theta1plus = theta1;
      theta1plus(i,j) += epsilon;
      theta1minus = theta1;
      theta1minus(i,j) -= epsilon;
      App1(i,j) = (Cost(theta1plus, theta2, theta3, X,y, lambda) - Cost(theta1minus,theta2, theta3, X,y,lambda)) / (2 * epsilon);
     end
  end
   
  for i = [1:m2];
    for j =[1:n2];
      theta2plus = theta2;
      theta2plus(i,j) += epsilon;
      theta2minus = theta2;
      theta2minus(i,j) -= epsilon;
      App2(i,j) = (Cost(theta1, theta2plus, theta3, X,y, lambda) - Cost(theta1,theta2minus, theta3, X,y,lambda)) / (2 * epsilon);
     end
  end
  
  for i = [1:m3];
    for j =[1:n3];
      theta3plus = theta3;
      theta3plus(i,j) += epsilon;
      theta3minus = theta3;
      theta3minus(i,j) -= epsilon;
      App3(i,j) = (Cost(theta1, theta2, theta3plus, X,y, lambda) - Cost(theta1,theta2, theta3minus, X,y,lambda)) / (2 * epsilon);
     end
  end
  
end
  