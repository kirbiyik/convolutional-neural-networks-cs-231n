import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.         #
    # Store the loss in loss and the gradient in dW. If you are not careful         #
    # here, it is easy to run into numeric instability. Don't forget the                #
    # regularization!                                                                                                                     #
    #############################################################################
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]

    for i in range(N):
        z = np.dot(X[i], W) # C size vector 
        p = np.exp(z)
        denom = np.sum(p)
        p = p / denom

        for j in range(C):
            # gradient
            y_one_hot = np.zeros((C)) 
            y_one_hot[y[i]] = 1
            dW[:, j] += (p[j] - y_one_hot[j]) * X[i]

        loss += -1 * np.log(p[y[i]])

    dW = (dW + reg) / N
    loss = loss / N  + reg / N * np.sum(np.abs(W))
    #############################################################################
    #                                                    END OF YOUR CODE                                                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    - W: (D, C) containing weights.
    - X: (N, D) containing a minibatch of data.
    - y: (N,) containing training labels; y[i] = c means


    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful         #
    # here, it is easy to run into numeric instability. Don't forget the                #
    # regularization!                                                                                                                     #
    #############################################################################
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]

    z = np.dot(X,W)  # (N, C)
    exp_z = np.exp(z)
    denom = exp_z.sum(axis=1) # stores sum of each rows
    # hadamard product, divide each row element by row sum for softmax
    denom = denom.reshape(-1,1)
    big_denom = np.tile(denom, (1,C))
    prob = np.multiply(exp_z, (1/big_denom))
    
    y_one_hot = np.zeros((N, C)) 
    y_one_hot[np.arange(N), y] = 1

    cost_matrix = -1 * np.log(np.sum(np.multiply(prob, y_one_hot), axis=1))

    dW = (np.dot(X.T, (prob - y_one_hot)) + reg ) / N
    # without regularization 
    loss = np.sum(cost_matrix) / N 
    # L1
    #loss +=  reg / N * np.sum(W)
    # L2
    loss += reg / N / 2 * np.sum(np.square(W))
    #############################################################################
    #                                                    END OF YOUR CODE                                                                 #
    #############################################################################

    return loss, dW

