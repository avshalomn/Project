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
class complex_weights_layer(tf.keras.layers.Layer)
  def __init__(self,shape,units=None):
    super(complex_weights_layer,self).__init__()
    self.shape = shape
    self.units = shape[0]*shape[1]
    self.W = make_tensor_weights()
    self.a = tf.Variable(tf.math.real(self.W), trainable = True)
    self.b = tf.Variable(tf.math.imag(self.W), trainable = True)
    
    
  def get_config():
    return super().get_config()
 
  def make_tensor_weights():
    def make_a_weight():
      a = b = 1
      while cmath.polar(a + 1j*b)[0] >= 1:
        a = np.random.random()
        b = np.random.random()
      return a +1j*b
    W = np.array([[make_a_weight()for c in range(self.shape[1])] for r in range(self.shape[0])]
    return W
