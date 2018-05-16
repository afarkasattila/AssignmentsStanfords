import numpy as np
from random import shuffle

"""
  Structured SVM loss function, naive implementation (with loops).

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
def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # compute the loss and the gradient
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        nr_of_corr_margin = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                nr_of_corr_margin += 1
                dW.T[j] += X[i]
                loss += margin
        dW.T[y[i]] += (-1) *  nr_of_corr_margin * X[i]
      # Right now the loss is a sum over all training examples, but we want it
      # to be an average instead so we divide by num_train.
    loss /= num_train
      # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg
    dW /= num_train

    num_dimensions = W.shape[0]
    dW = np.zeros(W.shape)
    for i in range(num_dimensions):
        for j in range(num_classes):
            for k in range(num_train):
                dW[i,j] += x[i]

    return loss, dW

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################




def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_pixel = X.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    scores = np.dot(X, W)
    subtr = (scores.T - scores[np.arange(num_train), y]).T + 1.0
    margins = np.maximum(0.0, subtr)
    margins[np.arange(num_train), y] = 0.0
    Li = np.sum(margins, axis=1)
    loss = np.average(Li)
    loss += reg * np.sum(W * W)

    pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

    nr_of_not_zeros = np.count_nonzero(margins, axis=1)

    for i in np.arange(num_train):
        dW.T[y[i]] += (X[i].T * nr_of_not_zeros[i]).T * (-1)
     
    indexes = np.argwhere(margins != 0)
    for i in np.arange(indexes.shape[0]):
        dW.T[indexes.T[1][i]] += X[indexes.T[0][i]]

    dW += 2 * reg
    dW /= num_train
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
    pass
    return loss, dW
