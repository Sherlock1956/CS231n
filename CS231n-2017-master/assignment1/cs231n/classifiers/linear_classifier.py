from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *



class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self,X,y,alpha,reg,iteration,batch_size = 200,verbose=False):
    num_train = X.shape[0]
    dim = X.shape[1]
    X_indicies = np.arange(num_train)
    num_classes = np.max(y)+1
    loss_his = []
    if self.W == None:
        self.W = 0.001*np.random.randn(dim,num_classes)
    for it in range(iteration):
        random_indicie = np.random.choice(X_indicies,batch_size)
        X_batch = X[random_indicie]
        y_batch = y[random_indicie]
        loss,grad = self.loss(X_batch,y_batch,reg)
        loss_his.append(loss)
        self.W += -(grad*alpha)
        if verbose and it % 100 == 0:
            print(f'itrations:{it},lose:{loss}')
    return loss_his
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

    pred_scores = np.dot(X,self.W)
    y_pred = np.argmax(pred_scores, axis=1)

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
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

