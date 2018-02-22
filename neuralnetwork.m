data = csvread('trainmod.csv');
data([1],:) = [];
data(:,[1]) = [];
[m n] = size(data);
y = data(:,[1]);
X = data(:,[2:n]);
X = [ones(m,1), X];

