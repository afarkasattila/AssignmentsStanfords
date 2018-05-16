import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i] - X[y[i]] ##

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  dW /= num_train

  #print (dW)

  dW += 2 * W

  #print ("\n", dW)

  #print ("\n", W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  correct_class_scores = scores[np.arange(len(y)), y]

  print ("X = ", X.shape)
  print ("W = ", W.shape)
  print (scores.shape)
  print ((correct_class_scores.reshape(500,1)).shape)

  idx = 1
  print ("Bemenet indexe (kep - X):", idx)
  print ("A bemeneti kep helyes osztalyanak indexe:", y[idx])
  print ("A bemeneti kep s-ei (scores):", scores[idx, :])
  print ("A bemeneti kep helyes s-e", correct_class_scores[idx])
  s = scores - correct_class_scores.reshape(500,1)
  print ("Elso ideiglenes tomb", s[idx])
  s[s < -1] = 0
  print ("Elso ideiglenes tomb nullazva", s[idx])
  Li = np.sum(s, axis=1)
  print ("Elso reszlegesen osszesitett veszteseg a bemeneti kep eseteben", Li[idx])
  Li += (num_classes - 1)
  print ("Osztalyok szama", (num_classes - 1))
  print ("Elso osszesitett veszteseg a bemeneti kep eseteben", Li[idx])
  print ("Elso osszesitett veszteseg", np.sum(Li)/ num_train)

  print ("Li.shape", Li.shape)

  print(W.shape)
  print((W**2).shape)
  print(np.sum(W).shape)
  print((np.sum(W**2)).shape)
  #print(sum)

  loss = np.sum(Li) / num_train + reg * np.sum(W*W)

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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
