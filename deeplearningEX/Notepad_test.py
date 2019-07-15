import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

from Part2.Optimization.opt_utils import load_params_and_grads, initialize_parameters, \
                                            forward_propagation, backward_propagation
from Part2.Optimization.opt_utils import compute_cost, predict, predict_dec, \
                                            plot_decision_boundary, load_dataset
from Part2.Optimization.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def update_parameterswith_gd(parameters, grads, learning_rate):
	L = len(parameters) // 2
	
	for l in range(L):
		parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
		parameters['b' + str(l+1)] -= learning_rate * grads['b' + str(l+1)]
	
	return parameters
	
	
def random_mini_batches(X, Y, batch_size=64, seed=0):
	m = X.shape[1]
	permuation = np.random.permuation(m)
	X_shuffle = X[:, permuation]
	Y_shuffle = Y[:, permuation]
	
	batch_num = int(m/batch_size)
	
	mini_batchs = []
	for i in range(batch_num):
		mini_batch_x = X_shuffle[:, i*batch_size : (i+1)*batch_size]
		mini_batch_y = Y_shuffle[:, i*batch_size : (i+1)*btach_size]
		mini_batch = (mini_batch_x, mini_batch_y)
		mini_batchs.append(mini_batch)
	
	if m % batch_num != 0:
		mini_batch_x = X_shuffle[:, batch_num*batch_size:]
		mini_batch_y = Y_shuffle[:, batch_num*batch_size:]
		mini_batch = (mini_batch_x, mini_batch_y)
		mini_batchs.append(mini_batch)
	
	
	
	
	
	
	
	
	



