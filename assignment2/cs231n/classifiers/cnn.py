import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               conv_param={'stride': 1}, 
               pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2}, 
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # (N,Ch,H,W)->conv->(N,F,Hc,Wc)->rl
    # ->mp->(N,F,Hp,Wp)
    # ->af:*(F*Hp*Wp,hd)->rl->af:*(hd,Cl)->(N,Cl)
    
    # init params of conv
    Ch, H, W, F = input_dim[0], input_dim[1], input_dim[2], num_filters
    self.params['W1'] = weight_scale * np.random.randn(F, Ch, filter_size, filter_size)
    self.params['b1'] = np.zeros(F)
    
    # compute param sizes of 1st affine, size after conv
    self.conv_param = {'stride': conv_param.get('stride', 1), 
                       'pad': conv_param.get('pad', (filter_size-1) / 2)}
    cstride, cpad = self.conv_param['stride'], self.conv_param['pad']
    Hc = (H+2*cpad-filter_size) / cstride + 1 # height after conv
    Wc = (W+2*cpad-filter_size) / cstride + 1 # weight after conv
    # size after pool
    self.pool_param = pool_param
    ph, pw, pstride = pool_param['pool_height'], pool_param['pool_width'], \
                      pool_param['stride']
    Hp = (Hc-ph) / pstride + 1 # height after pool
    Wp = (Wc-pw) / pstride + 1 # weight after pool
    # init params of 1st affine
    self.params['W2'] = weight_scale * np.random.randn(F*Hp*Wp, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    
    # init params of hidden layer
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    # filter_size = W1.shape[2]
    conv_param = self.conv_param
    # 
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = self.pool_param
    # I think parameters above should initialized in __init__(), not here.

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    caches = []
    reg = self.reg

    # (N,Ch,H,W)->conv->(N,F,Hc,Wc)->rl->mp->(N,F,Hp,Wp)
    out, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    caches.append(cache)
    # ->af:*(F*Hp*Wp,hd)->rl->(F*Hp*Wp,hd)
    out, cache = affine_relu_forward(out, W2, b2)
    caches.append(cache)
    # ->af:*(hd,Cl)->(N,Cl)->softmax->(N,Cl)
    scores, cache = affine_forward(out, W3, b3)
    caches.append(cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += np.sum(0.5*reg*np.sum(w) for w in [W1, W2, W3])
    
    dout, dW3, db3 = affine_backward(dout, caches.pop())
    dout, dW2, db2 = affine_relu_backward(dout, caches.pop())
    _, dW1, db1 = conv_relu_pool_backward(dout, caches.pop())
    
    grads['W1'], grads['b1'] = dW1+reg*W1, db1                  # regularization
    grads['W2'], grads['b2'] = dW2+reg*W2, db2                  # regularization
    grads['W3'], grads['b3'] = dW3+reg*W3, db3                  # regularization
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
