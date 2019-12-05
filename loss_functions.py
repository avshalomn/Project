import tensorflow as tf



@tf.function
def phase_loss_func(y_pred,y):
  """
  
  assuming y_pred and y are of the shape a +jb
  phase = tf.math.atan2(b/a)
  amp = tf.math.sqrt(a^2 + b^2)
  """
  pred_a,y_a = tf.math.real(y_pred),tf.math.real(y)
  pred_b,y_b = tf.math.imag(y_pred),tf.math.imag(y)
  
  pred_phase = tf.math.atan2(pred_b,pred_a)
  y_phase = tf.math.atan2(y_b,y_a)
  
  loss = 0.5 * (tf.math.abs(pred_phase - y_phase))**2   # assuming the orginal weights/phases start at 0 per pixel
  
  loss = tf.cast(loss,tf.float32)
    
  return loss

###################################################################

@tf.function
def complex_loss_func(y_pred,y):
  """
  Assuming weights are complex!
  w = wa +jwb
  
  """
  pred_phase = tf.math.atan2(tf.math.imag(y_pred),tf.math.real(y_pred))
  y_phase = tf.math.atan2(tf.math.imag(y),tf.math.real(y))
  
  diff = tf.abs(y_phase - pred_phase)
  
   
  return loss

####################################################################

"""
NEW IDEA! is it possible to do better if a layer hold 2 setes of weights, a & b and within thc call() of the layer,
and within the loss/training loop!
"""
@tf.function
def new_loss_func(y_pred,y):
  """
  y_pred: form of a +jb
  y: form if a+jb
  This is for the case where the layer has two different weights - a & b
  """
  main_diff = y_pred - y
  
  phase_diff = tf.math.atan2(
    tf.math.imag(main_diff),
    tf.math.real(main_diff)
  )
  
  js = tf.constant([[1j for c in y_pred.shape[1]] for r in y_pred.shape[0]])
  
  phase_loss = phase_diff*js
  
  return phase_loss
  
  

