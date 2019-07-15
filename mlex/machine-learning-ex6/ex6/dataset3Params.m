function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possible_value = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
p_m = size(possible_value);

select_C = 0.01;
select_sigma = 0.01;
error_rate = 1;

for i = 1:p_m
	C = possible_value(i);
	for j = 1:p_m
		sigma = possible_value(j);	
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		
		cur_error_rate = mean(double(pred ~= yval));
		if(cur_error_rate < error_rate)
			error_rate = cur_error_rate;
			select_C = C;
			select_sigma = sigma;
		end
	end
end

C = select_C;
sigma = select_sigma;


% =========================================================================

end
