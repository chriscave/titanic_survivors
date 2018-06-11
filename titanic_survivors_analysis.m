clear ; close all; clc
fprintf('Here we will analyse the titanic survival database found on kaggle. We wil do this by using deep neural netowrks. This is more of an exercise in building neural networks than scoring high accuracy. \n')

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Loading data....This database is based off the kaggle post found at https://www.kaggle.com/niklasdonges/end-to-end-project-with-python \n')
data = csvread('trainmod.csv');

rand("seed",42) #makes sure we keep the same training and CV set.
data = data(randperm(size(data, 1)), :); 

data([1],:) = []; #removes some of the zeros in the data
data(:,[1]) = [];
[m n] = size(data);

data = [ones(m,1),data]; #adding the bias nodes
data(:,[1,2]) = data(:,[2,1]);

#splits data into training, cv and test set
data_train_cv = data((1:(ceil(0.8* m))),:); 
data_test_set = data((ceil(0.8 * m) +1 :end), :);

[m n] = size(data_train_cv);

# here we allow for random permutations in the training and cv set
rand('seed','reset'); 
data_train_cv = data_train_cv(randperm(size(data_train_cv, 1)), :); 

#splitting into training data and labels
X_train = data_train_cv((1:(ceil(0.75* m))),[2:end]); #0.75 is used here because the cv set is meant to be the same 20% of the whole data set.
X_cv = data_train_cv((ceil(0.75 *m) +1 : end), [2:end]);
X_test = data_test_set(:, [2:end]);

y_train = data_train_cv((1:(ceil(0.75* m))),[1]);
y_cv = data_train_cv((ceil(0.75 *m) +1 : end), [1]);
y_test = data_test_set(:, [1]);

y = data(:,[1]);
X = data(:,[2:end]);
[x1 x2] = size(X);

fprintf('First some quick analysis, what is the percentage of people in the training data that survived? \n')

fprintf('Program paused. Press enter to continue.\n');
pause;

survived = sum(y(:) == 1);
survived_percent = (survived / (size(y)(1))) * 100; #this gives the percentage of people that survived

fprintf(['Percentage of people survived: %f \n'], survived_percent)
fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('What is the baseline accuracy? What if we predicted that every woman survived? \n')
fprintf('Program paused. Press enter to continue.\n');
pause;

y_female_survived = X(:,3); #in the data female = 1 and male = 0
surv_index = find(y);
dead_index = find(~y);
CM = zeros(2);

#finding the confusion matrix if we predict every woman survives and no men survive.
A = y_female_survived(surv_index);
CM(1,1) = sum(A);
CM(1,2) = length(A) - sum(A);

B = y_female_survived(dead_index);
CM(2,1) = sum(B);
CM(2,2) = length(B) - sum(B);

fprintf('Here is the confusion matrix.\n')
CM
base_acc =   (CM(1,1) + CM(2,2)) / sum(CM(:));
fprintf(['Baseline accuracy is : %f \n'] , base_acc)

fprintf('We hope our neural network can achieve higher accuracy than this \n')
#the number of nodes and hidden layers in the neural network.
hl = input("We will analyse a neural network. Pick the amount of hidden layers and nodes by entering a 1 x n row vector where n is the amount of hidden layers and each entry corresponds to the amount of nodes in that layer \n");
layer = [(x2-1), hl, 1];

fprintf('We first initialise the weights of the neural network \n');

epsilon = zeros(1, size(hl,2) +1);

#row counters and column counters are crucial in the flexibility of this analysis.
#They show where the block matrices in theta start.
#If theta is n x m matrix then the final value of rowct and colct respectively is (n+1) and (m+1).
#This helps with the for loops.

rowct = zeros(1, size(hl,2) + 1); 
colct = zeros(1, size(hl, 2) + 1);
rowct(1) = 1;
colct(1) = 1;
rowct = [rowct,sum(hl) + 2];
colct = [colct,  x2 + sum(hl) + size(hl,2)+1];

for i = 1: (size(rowct,2) -1)
  epsilon(i) = sqrt(6) / (sqrt(layer(i+1)) + sqrt(layer(i)));
end
for i = 2: (size(rowct,2) -1);
  colct(i) = colct(i-1) + layer(i-1) + 1;
  rowct(i) = rowct(i-1) + layer(i);
end
rand('seed','reset');
init_theta = [];
for i = 1:(size(rowct,2) -1);
  init_theta_ = rand(layer(i+1), layer(i) + 1)* (2* epsilon(i)) - epsilon(i);
  init_theta = blkdiag(init_theta,init_theta_);
end
#This is used to check the backpropagation algorithm is correct.
#fprintf('Now we check that the backpropagation algorithm is working. The first matrix is the derivative and the second matrix is an approximation. They should be very close to each other.\n');
#fprintf('Program paused. Press enter to continue.\n');
#pause;
#epsilon = 0.001;
#lambda = input('Enter a value for lambda to test the backpropagation algorithm. ');
#D = BackProp(X_train,y_train, init_theta, lambda, rowct,colct)
#D_aprox = gradientchecking(X_train,y_train,init_theta,lambda,epsilon,rowct,colct)


#Usually num_iters = 1500
num_iters = input('We will now perform gradient descent. How many iterations shall we perform the descent? ');
alpha = input('Enter in a series of learning rates to test out. Here lambda will be set to 0 initially. ');
stralpha = [];
Cost_history=[];
for i = 1: size(alpha,2);
  [A,B] = gradientDescent(X_train,y_train,init_theta,0,alpha(1,i),num_iters,rowct,colct);
  stralpha = [stralpha; strcat("alpha = ", num2str(alpha(1,i)))];
  Cost_history = [Cost_history,B];
end
#this plots learning rates against cost, where lambda is set to 0 and performing num_iters amount of iterations.
plot((1:num_iters),Cost_history);
xlabel('Number of iterations');
ylabel('Cost');
c = cellstr(stralpha); #this provides the legend
legend(c);

#read this from the graph. Usually 0.3 is a good bet.
alpha = input('What is the optimal choice of alpha? ');
fprintf('Now we plot a validation curve to find the optimal value of lambda. ');

#This plots how the error in the training set and the cross validation set behaves with choices of lambda.
#Lambda_vect = [0 0.001 0.003 0.01 0.03 0.1 0.3 0.5 1 3 5 10]. Ths can be changed in the ValidationCurve function.

[lambda_vect, error_train, error_cv] =  validationCurve(X_train, y_train, X_cv, y_cv, init_theta, alpha, num_iters, rowct, colct);
plot(lambda_vect,error_train,lambda_vect,error_cv);
xlabel('lambda');
ylabel('Cost');
str = ["training set"; "CV set"];
c = cellstr(str);
legend(c);

#Read this from the graph
lambda = input('What is the optimal value of lambda? ');

fprintf('Now we plot a learning curve for this value of lambda.\n');



#This plots how the cost behaves after increasing the size of the training set. This has been modified to only perform 50 gradient descents.
#You can change this in the learningCurve function.
#This should give you an indication whether the neural network is overfitting or underfitting.

[error_train, error_cv] = learningCurve(X_train, y_train, X_cv, y_cv, lambda, init_theta, alpha, num_iters, rowct, colct);
plot(1:length(error_train), error_train, 1:length(error_train), error_cv);
xlabel('Examples');
ylabel('Cost');
str = ["training set"; "CV set"];
c = cellstr(str);
legend(c);
fprintf('Program paused. Press enter to continue.\n');
pause;

#This trains the network using the optimal alpha and lambda that has just been found.
[theta, Cost_history] = gradientDescent(X_train, y_train,init_theta,lambda,alpha,num_iters,rowct,colct);
fprintf('Here are the probability of predicted values of the labels on the cross validation set\n');
H = hyp(X_cv, theta, rowct,colct);
y_cv_prob = H(end,:)
fprintf('Program paused. Press enter to continue.\n');
pause;

#This provides the confusion matrix for threshold = 0.5
fprintf('Now we find the confusion matrix when the threshold is 0.5.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;
threshold = 0.5;
CM = confusion_matrix(X_cv,y_cv,threshold,theta,rowct,colct)
fprintf('This has accuracy, precision, recall and F score:\n');
accuracy = (CM(1,1) + CM(2,2)) / sum(CM(:))
[precision, recall, F_score] = precision_recall(X_cv,y_cv,threshold,theta,rowct,colct)

fprintf('We shall now plot a precision, recall curve.\n')
fprintf('Program paused. Press enter to continue.\n');
pause;

H = hyp(X_cv, theta, rowct,colct);
y_cv_prob = H(end,:); 
#this gives suitable tests for threshold for precision and recall
max_thres = max(y_cv_prob);
min_thres = min(y_cv_prob);
threshold = linspace(min_thres,max_thres,50);
precision = [];
recall = [];
f_score = [];
for i = threshold;
  [A,B,C] = precision_recall(X_cv,y_cv,i,theta,rowct,colct);
  precision = [precision, A];
  recall = [recall, B];
end
#sometimes we can end up dividing by zero. This is corresponds to when precision = 1.
#this is why we compute the f-score here instead inside the for loop.
precision(isnan(precision)) = 1;
f_score = 2 * ((precision .* recall) ./ (precision + recall));

#plotting precision-recall and f-score recall curves.
ax = plotyy(recall, precision, recall, f_score);
xlabel('Recall');
ylabel(ax(1), 'Precision');
ylabel(ax(2), 'F-score');
fprintf('We shall now plot accuarcy against the threshold.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

#Now finding and plotting the accuarcy for different thresholds.
accuarcy = [];
for i = threshold;
  CM = confusion_matrix(X_cv,y_cv,i,theta,rowct,colct);
  accuarcy_ = (CM(1,1) + CM(2,2)) / sum(CM(:));
  accuarcy = [accuarcy, accuarcy_];
end
plot(threshold, accuarcy);
xlabel('Threshold');
ylabel('Accuarcy');

#Read this from the graph.
threshold = input('What threshold gives the best accuarcy? ');
CM = confusion_matrix(X_test,y_test,threshold,theta,rowct,colct);
accuarcy = (CM(1,1) + CM(2,2)) / sum(CM(:));
fprintf(['So the accuarcy on the test set is: %f\n'],accuarcy); #this is the score of our neural network

#This is so we can remember these results and plot them

M = [accuarcy, lambda, alpha, sum(hl),length(hl), threshold, hl];
dlmwrite('Accuarcy_results.csv',M,"-append"); #this writes the results to this following file
acc_results = csvread('Accuarcy_results.csv');
X = acc_results(:,5);
Y = acc_results(:,4);
Z = acc_results(:,1);
scatter3(X,Y,Z);
xlabel('Hidden layers');
ylabel('Amount of nodes');
zlabel('Accuracy');
hold on

#this plots the plane z = base_acc. If the points we plot is above this plane then the accuarcy is better than the baseline accuracy.
[x,y] = meshgrid(X,Y);
z = base_acc * ones(size(x));
surf(x,y,z);
#The labels are in the following order: choice of lambda, choice of learning rate, choice of threshold, architecture of the neural network. 
label = acc_results(:,[2,3]);
label = [label, acc_results(:,(6:end))];
label = num2str(label);
label = cellstr(label);
dx = 0.02;
dy = 0.2;
dz = 0.0002;
text(X + dx, Y + dy, Z + dz, label);

hold off

#fprintf('We will load the test data and provide our predictions.\n');
#data_test = csvread('testmod.csv');
#data_test([1],:) = [];
#data_test(:,[1]) =[];
#P_id = data_test(:,1);
#data_test(:,1) = [];

#[k l] = size(data_test);
#test = [ones(k,1),data_test];


#H = hyp(test, theta, rowct,colct);
#y_test_prob = H(end,:);
#y_test_pred = zeros(1,k);
#survive = find(y_test_prob > threshold);
#y_test_pred(survive) = 1;
#died = find (y_test_prob <= threshold);
#y_test_pred(died) = 0;



#prediction = [P_id, y_test_pred'];
#csvwrite('predictions.csv', prediction);

