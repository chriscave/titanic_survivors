data = csvread('trainmod.csv');
data([1],:) = [];
data(:,[1]) = [];
[m n] = size(data);
y = data(:,[1]);
X = data(:,[2:n]);
X = [ones(m,1), X];
#Removing features that do not affect results that much.
X(:,[6]) = [];
X(:,[9]) = [];
[x1 x2] = size(X);

s2 = ceil((x2 - 1) / 2);
s3 = floor((x2 - 1) / 2);
#s2 = 11;
#s3 = 11;

#random initialisation
epsilon1 = sqrt(6) / (sqrt(s2) + sqrt(x2));
epsilon2 = sqrt(6) / (sqrt(s3) + sqrt(s2));
epsilon3 = sqrt(6) / (1 + sqrt(s3));

theta1 = rand(s2, x2)* (2* epsilon1) - epsilon1;
theta2 = rand(s3, s2 + 1) * (2*epsilon2) - epsilon2;
theta3 = rand(1, s3 +1) * (2*epsilon3) - epsilon3;

[m1 n1] = size(theta1);
[m2 n2] = size(theta2);
[m3 n3] = size(theta3);

#gradientchecking
#epsilon = 1e-4
#[App1,App2, App3] = gradientchecking(theta1, theta2, theta3, X,y,lambda,epsilon);

#[D1,D2,D3] = BackProp(X,y,theta1,theta2,theta3,lambda);

#D1 - App1;
#D2 - App2;
#D3 - App3;


alpha = 0.3;
lambda = 0.03;
num_iters = 4000;

[CVprob, mn, s] = crossVal(X,y,theta1, theta2, theta3,alpha, lambda, num_iters);
[prob,theta, Cost_history, J] = nn(theta1, theta2, theta3, alpha, lambda, num_iters, X,y);