import numpy as np
from random import shuffle

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
  num_class = W.shape[1]
  num_train = X.shape[0]
  dW_i = np.zeros_like(dW)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)                   # scores with normalization logC
    score_yi = scores[y[i]]
    
    loss -= score_yi
    sum_exp = 0
    for j in xrange(num_class):
      if j == y[i]:
        sum_exp += np.e**score_yi
        continue
      exp_score = np.e**scores[j]
      dW_i[:, j] = exp_score*X[i]
      sum_exp += exp_score
    loss += np.log(sum_exp)
    dW_i[:, y[i]] = -(sum_exp-np.e**score_yi)*X[i]
    # divide by the sum of exp for current data point all at once
    dW_i /= sum_exp
    dW += dW_i
  loss = (loss/num_train) + 0.5*reg*np.sum(W*W)
  dW = (dW/num_train) + reg*W
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
  num_class = W.shape[1]
  num_train = X.shape[0]
  Score = X.dot(W)
  Score = Score - Score.max(axis=1, keepdims=True)
  Exp_score = np.exp(Score)
  Probability = Exp_score / Exp_score.dot(np.ones([num_class, 1]))
  Mask_y = np.zeros_like(Score)
  Mask_y[xrange(num_train), y] = 1.
  Mask_log_pb = Mask_y * np.log(Probability)
  loss = -np.sum(Mask_log_pb) / num_train + 0.5*reg*np.sum(W*W)
  
  Grad_pb = -Probability
  Grad_pb[xrange(num_train), y] += 1
  dW = -X.T.dot(Grad_pb) / num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

