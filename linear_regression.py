# Daniel Eftekhari
# Correspondences to daniel.eftekhari@mail.utoronto.ca

import numpy as np
import matplotlib.pyplot as plt

'''
A simple linear regression module with L2 loss
'''

def eval_loss(X, w, y, alpha):
    return 0.5*np.sum(np.power((np.dot(X,w) - y),2)) + 0.5*alpha*np.dot(np.transpose(w),w)

def grad_loss(X, w, y, alpha):
    return np.dot(np.transpose(X),(np.dot(X,w) - y)) + alpha*w

def shuffle(X, y):
    assert len(X) == len(y)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

if __name__ == "__main__":
    try:
        # load data set
        from sklearn.datasets import load_boston
        dataset = load_boston()
        X, y = dataset.data, dataset.target
    except:
        print('Please load your own data set in. X should have shape N x M and y shape N x 1, where N is the number of samples and M nunber of features')

    # randomly shuffle data
    X, y = shuffle(X, y)

    # reshape y into (N x 1)
    y = np.reshape(y, (y.shape[0], 1))

    # initialize w using a normal distribution with zero mean and unit variance
    weights = np.random.normal(0, 1, (X.shape[1], 1))

    # hyperparameters
    list_lambda = np.asarray([0.001, 0.01, 0.1, 0.9]) # learning rate
    list_alpha = np.asarray([0, 0.01, 0.1, 0.9]) # regularization

    # Split into training, validation, testing sets
    # For simplicity, we won't do nested k-fold cross validation (although it would help maximize data utility)
    X_training = X[0:int(X.shape[0] * 3 / 5.0)]
    X_validation = X[int(X.shape[0] * 3 / 5.0):int(X.shape[0] * 4 / 5.0)]
    X_testing = X[int(X.shape[0] * 4 / 5.0):]
    y_training = y[0:int(y.shape[0] * 3 / 5.0)]
    y_validation = y[int(y.shape[0] * 3 / 5.0):int(y.shape[0] * 4 / 5.0)]
    y_testing = y[int(y.shape[0] * 4 / 5.0):]

    # Normalization using mean and standard deviation of training set
    mean = np.mean(X_training, axis=0)
    std = np.std(X_training, axis=0)
    X_training = (X_training - mean) / std
    X_validation = (X_validation - mean) / std
    X_testing = (X_testing - mean) / std

    # We will only tune lambda and alpha
    # We will not tune for the batch size (we use full training set)
    # We will not tune for the number of epochs (we use 100 epochs)
    num_epochs = 35

    # Housekeeping variables
    val_loss = np.zeros((list_lambda.shape[0], list_alpha.shape[0]))
    list_w = np.zeros((list_lambda.shape[0], list_alpha.shape[0], X.shape[1], 1))
    loss = np.zeros((list_lambda.shape[0], list_alpha.shape[0], num_epochs-1))
    epochs = np.linspace(1, num_epochs, num_epochs, dtype=np.uint16)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    print('Initial test set loss')
    for i in range(list_alpha.shape[0]):
        print('loss is ' + str(eval_loss(X_testing, weights, y_testing, list_alpha[i])) + ' with regularization ' + str(list_alpha[i]))

    # run gradient descent algorithm
    for i in range(list_lambda.shape[0]):
        for j in range(list_alpha.shape[0]):
            plt.suptitle('loss vs epochs for learning rate = ' + str(list_lambda[i]) + ' and regularization = ' + str(list_alpha[j]))
            w = np.zeros((X.shape[1], 1)) + weights
            for k in range(1, num_epochs):
                w = w - list_lambda[i] * grad_loss(X_training, w, y_training, list_alpha[j])
                loss[i,j,k-1] = eval_loss(X_validation, w, y_validation, list_alpha[j])
                if k < num_epochs - 1:
                    plt.plot(epochs[0:k], loss[i,j,0:k], color='b')
                    plt.pause(0.002)
                else:
                    plt.show()
            print(loss[i,j,-1])
            val_loss[i,j] = loss[i,j,-1]
            list_w[i,j] = w

    print('Validation loss matrix w.r.t hyperparameters')
    print(val_loss)
    index = np.argmin(val_loss)
    lambda_index = int(index/val_loss.shape[1])
    alpha_index = index - lambda_index * val_loss.shape[1]
    optimal_lambda, optimal_alpha = list_lambda[lambda_index], list_alpha[alpha_index]
    print('optimal learning rate, optimal regularization')
    print(optimal_lambda, optimal_alpha)
    optimal_w = list_w[lambda_index, alpha_index]

    print('Optimal test set loss')
    print('loss is ' + str(eval_loss(X_testing, optimal_w, y_testing, optimal_alpha)) + ' with regularization ' + str(optimal_alpha))