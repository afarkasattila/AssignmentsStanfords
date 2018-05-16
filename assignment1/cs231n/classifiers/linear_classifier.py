from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers.linear_svm import svm_loss_naive
#from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
          # lazily initialize W
          self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
            indexes = np.arange(num_train)
            np.random.shuffle(indexes)
            indexes =  indexes[:batch_size]
            X_batch = X[indexes] # sample 256 examples
            y_batch = y[indexes]
            l, dW = svm_loss_naive(self.W, X_batch, y_batch, reg)
            #from cs231n.gradient_check import grad_check_sparse
            #loss1, grad1 = svm_loss_naive(self.W, X_batch, y_batch, 5e1)
            #f = lambda w: svm_loss_naive(w, X_batch, y_batch, 5e1)[0]
            #grad_numerical = grad_check_sparse(f, self.W, grad1)
            if it % 100==0 and verbose:
                print("%d : %f"%(it,l))
            loss_history.append(l)
            self.W += - learning_rate * dW # perform parameter update
        #l, dW = svm_loss_vectorized(self.W, X_batch, y_batch, reg)
      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################


        return loss_history
    
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = np.dot(X, self.W)
        #print(scores.shape)
        #print(scores)
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred
    
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        dW = np.zeros(self.W.shape) # initialize the gradient as zero
        # compute the loss and the gradient

        num_classes = self.W.shape[1]
        num_train = X_batch.shape[0]
        loss = 0.0

        for i in range(num_train):
            nr_of_corr_margin = 0
            scores = X_batch[i].dot(self.W)
            correct_class_score = scores[y_batch[i]]
            for j in range(num_classes):
                if j == y_batch[i]:
                    continue
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                if margin > 0:
                    nr_of_corr_margin += 1
                    dW.T[j] += X_batch[i]
                    loss += margin
            dW.T[y_batch[i]] += (-1) *  nr_of_corr_margin * X_batch[i]
          # Right now the loss is a sum over all training examples, but we want it
          # to be an average instead so we divide by num_train.
        loss /= num_train
          # Add regularization to the loss.
        loss += reg * np.sum(self.W * self.W)
        dW += 2 * reg
        dW /= num_train
        return loss, dW
    
class LinearSVM(LinearClassifier):
      """ A subclass that uses the Multiclass SVM loss function """
#def loss(self, X_batch, y_batch, reg):
 #   return svm_loss_naive(self, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
