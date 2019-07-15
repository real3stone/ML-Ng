%% Ex3
%% This is just for test!

clear; close all; clc

input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

%%----Part 1: Loading and Visualizing Data ----
load('ex2data1.mat');
m = size(X, 1);

	% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

%%----Part 2: Loading Parameters ----
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];

%%----Part 3&4: Compute Cost (Feedforward) ----
lambda = 1; % with regularization
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,...
					num_labels, X, y, lambda);
					
%%----Part 5: Sigmoid Gradient ----
g = sigmoidGradient([-1 -0.5 0 0.5 1]); 
fprintf('%f', g); % test

%%----Part 6: Initializing Pameters ----
initial_Theta1 = randInitialzeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitialzeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:); initial_Theta2(:)]; % Unroll

%%----Part 7: Implement Backpropagation
checkNNGradients;

%%----Part 7: Implement Regularization
lambda = 3;
checkNNGradients(lambda);

	% Also output the costFunction debugging values
debug_J = nnCostFunction(nn_params, input_layer_size, ...
                       hidden_layer_size, num_labels, X, y, lambda);

%%----Part 8: Training NN
options = optimset('MaxIter', 50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
									input_layer_size,
									hidden_layer_size,
									num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%%----Part 9: Visualize Weights
displayData(Theta1(:, 2:end));

%%----Part 10: Implement Predict
pred = predict(Theta1, Theta2, X);
fprintf('\nTrainin Set Accuracy: %f\n', mean(double(pred == y)) * 100);




  