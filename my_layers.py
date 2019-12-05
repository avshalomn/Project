import tensorflow as tf
import cmath
import numpy as np

class layer2D(tf.keras.layers.Layer):
  def __init__(self,shape,units = None):
    super(two2layer,self).__init__()
    self.shape = shape
    self.units = shape[0] * shape[1]
    
  def get_config():
    return super().get_config()
    
################################
##  double weights layer ##
################################    
class complex_weights_layer(tf.keras.layers.Layer)
  def __init__(self,shape,units=None):
    super(complex_weights_layer,self).__init__()
    self.shape = shape
    self.units = shape[0]*shape[1]
    self.W = make_tensor_weights()
    self.a = tf.Variable(tf.math.real(self.W), trainable = True)
    self.b = tf.Variable(tf.math.imag(self.W), trainable = True)
    
  def get_config(self):
    return super().get_config()
 

  def call(self,inputs):
    
    i_a = tf.math.real(inputs)
    i_b = tf.math.imag(inputs)
    # inputs amp and phase #
    i_amp = tf.math.sqrt(i_a**2 + i_b**2)  # inputs amp
    i_phase = tf.math.atan2(i_b,i_a)
    # weights amp and phase #
    w_amp = tf.math.sqrt(self.a**2 + self.b**2)
    w_phase = tf.math.atan2(self.b,self.a)
    
    
    
    
    return
  

  def make_tensor_weights(self):
    def make_a_weight():
      a = b = 1
      while cmath.polar(a + 1j*b)[0] >= 1:
        a = np.random.random()
        b = np.random.random()
      return a +1j*b
    W = np.array([[make_a_weight()for c in range(self.shape[1])] for r in range(self.shape[0])]
    return W
