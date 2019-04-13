%This is test.m


data = load('ex1data1.txt'); %注释

% first column: population %
% second column: profit

X = data(:, 1); % vectorization
y = data(:, 2);

m = length(y);   % number of training examples
figure;

plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

X = [ones(m, 1), X];  %Add a column of ones to x
theta = zeros(2, 1);

iterations = 1500;
alpha = 0.01;


J = sum((X * theta - y).^2) / (2 * m);  % cost function

