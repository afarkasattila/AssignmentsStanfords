import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  D = W.shape[0]
  C = W.shape[1]

  scores = np.dot(X, W) # (N,C)

  #print(enumerate(scores[0]))
  #print("W = ", W)
  print("D = ", D)
  print("C = ", C)
  print("N = ", N)

  L = np.zeros(N)

  for currentX in np.arange(N):

    numerator = np.exp(scores[currentX, y[currentX]])
    denominator = 0

    #for currentCindex, currentC in enumerate(scores[currentX]):
      #print (currentX)
      #print (ss)
      #print (scores[currentX])
      #print(y[currentX])
      #if (currentCindex != y[currentX]):
      #  denominator += np.exp(scores[currentX, y[currentX]])

      #L[currentCindex] = - np.log(numerator / denominator)

    scoreCorectClass = np.exp(scores[currentX, y[currentX]])
    scoreSum = np.sum(np.exp(scores[currentX]))

    L[currentX] = -np.log(scoreCorectClass / scoreSum)

    tdW = np.zeros_like(W)
    for a in np.arange(D):
      for b in np.arange(C):

        sb = np.exp(scores[currentX, b])

        if (b == y[currentX]):
          tdW[a, b] = (sb / scoreSum - 1) * X[currentX, a]
        else:
          tdW[a, b] = (sb / scoreSum) * X[currentX, a]

    dW += tdW / N

    #print (currentX, L[currentX])

  dW += reg * 2 * W

  loss += np.sum(L)/N + reg * np.sum(W**2)


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]

  idxN = np.arange(N)

  scores = np.dot(X, W) # (N,C)
  scores = np.exp(scores)

  correctScores = scores[idxN, y]
  scoreSum = np.sum(scores, axis=1)

  # Calculate the loss.
  loss = -np.log(correctScores / scoreSum)
  loss = np.sum(loss) / N + reg * np.sum(W ** 2)

  # Normalize the scores.
  scoreSum = np.reshape(scoreSum, (scoreSum.shape[0], 1))
  scores = scores / scoreSum

  # Substract 1 from the correct class scores.
  scores[idxN, y] = scores[idxN, y] - 1

  # Prepare the matrices so they can be broadcasted together.
  scores = scores.reshape((scores.shape[0], 1, scores.shape[1]))
  X2 = X.reshape((X.shape[0], X.shape[1], 1))

  # Broadcast the inputs to calculated and regularized scores.
  dW = scores * X2

  # Average the gradients.
  dW = np.sum(dW, axis=0) / N

  # Add regularization.
  dW += reg * 2 * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

