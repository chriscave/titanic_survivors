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

X = [X, X(:,10).^2]; #adding polynomial features for title. Third degee polynomial seems to be overfitting.


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

alpha = 0.2;
lambda = 0.06;
num_iters = 1000;

k = randperm(x1);
k = k';

sixty_percent = ceil(x1 * .6);
eighty_percent = ceil(x1 * .8);

X_train = X(k(1: sixty_percent),:);
X_cv = X(k((sixty_percent + 1) : eighty_percent),: );
X_test = X(k((eighty_percent + 1) : x1), :);
y_train = y(k(1: sixty_percent));
y_cv = y(k((sixty_percent + 1) : eighty_percent));
y_test = y(k((eighty_percent + 1) : x1));

[prob,theta, Cost_history, J] = nn(theta1, theta2, theta3, alpha, lambda, num_iters, X_train,y_train);

theta1 = theta([1:m1],[1:n1]);
theta2 = theta([m1 + 1: m1 + m2], [n1 + 1: n1 + n2]);
theta3 = theta([(m1 + m2 + 1): m1 + m2 + m3], [(n1 + n2 +1): (n1 + n2 + n3)]);

[a2,a3,a4] = hyp(X_cv,theta1,theta2,theta3);

[p,prob] = predict(theta1, theta2,theta3, X_cv,y_cv);
Err_analysis = [X_cv,p',y_cv];