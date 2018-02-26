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

#random initialisation
epsilon1 = sqrt(6) / (sqrt(x2 - 1) + sqrt(x2 - 1));
epsilon2 = sqrt(6) / (1 + sqrt(x2 - 1));
theta1 = rand(x2-1, x2)* (2* epsilon1) - epsilon1;
theta2 = rand(1,x2) * (2*epsilon2) - epsilon2;
[m1 n1] = size(theta1);
[m2 n2] = size(theta2);

alpha = 0.3;
lambda = 0.03;
num_iters = 4000;

#Cross validation of neural network
[CVprob, mn, s] = crossVal(X,y,theta1, theta2,alpha, lambda, num_iters);
[prob,theta, Cost_history, J] = nn(theta1, theta2, alpha, lambda, num_iters, X,y);

Theta1 = theta([1:m1],[1:n1]);
Theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);

testdata = csvread('testmod.csv');
Pid = testdata(:,[2]);
Pid([1]) = [];
testdata(:,[1,2])=[];
testdata([1],:) =[];

[a b] = size(testdata);
X_test = [ones(a,1), testdata];
X_test(:,[6]) = [];
X_test(:,[9]) = [];

[a2,a3] = hyp(X_test,Theta1, Theta2);
p = round(a3);
pred = [Pid, p'];


