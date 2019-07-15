import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def basic_sigmoid(x):
    s = 1.0 / (1 + 1/math.exp(x))  # only real number
    return s


def sigmoid(x):
    s = 1.0 / (1 + 1/np.exp(x))
    return s


def sigmoid_derivative(x):
    s = 1.0 / (1 + 1/np.exp(x))
    ds = s * (1 - s)
    return ds


def L1(yhat, y):
    loss = np.sum(np.abs(yhat - y))
    return loss


def L2(yhat, y):
    loss = np.sum(np.power((y - yhat), 2))
    return loss


# loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# orig shape (m, num_px, num_py, 3). e.g.(209, 64,64,3)
m_train = train_set_x_orig.shape[0]  # 209
m_test = test_set_x_orig.shape[0]   # 50
num_px = train_set_x_orig.shape[1]  # 64


# 预处理：变成一个长向量
''' A trick when you want to flatten a matrix X of shape (a,b,c,d) 
            to a matrix X_flatten of shape (b*c*d, a) is to use:

    X_flatten = X.reshape(X.shape[0], -1).T
'''
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# standardize：(range:0-255)
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# 初始化
def initialize_with_zeors(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


# 前向计算
def propagate(w, b, X, Y):
    """ Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
    m = X.shape[1]  # num of examples
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * (np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))

    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# 迭代训练
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)  # record cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)  # w本来不就是这种结构的吗？干嘛多此一举

    A = sigmoid(np.dot(w.T, X) + b)
    # Y_prediction = (A > 0.5) * 1      # 直接向量化(相乘时false会当做0运算)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1  # 所以Y_prediction是矩阵？？
        else:
            Y_prediction[0, i] = 0

    assert(Y_prediction.shape == (1, m))
    return Y_prediction


# 构造模型
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """ Builds the logistic regression model

    Arguments:
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeors(X_train.shape[0])
    # train
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    # predict
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # check
    print("train accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


# 运行
d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

# Plot learning curve
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('costs')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# try different learning rate
print('\n' + "-----------------------------------" + '\n')
learning_rates = [0.01, 0.001, 0.0001]
models = {}  # dictionary
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y,
                           num_iterations=1500, learning_rate=i, print_cost=False)
    print('\n' + "-----------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations(hundreds)')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


