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
  
  new_phase = 
  
  return loss

####################################################################

@tf.function
"""
NEW IDEA! is it possible to do better if a layer hold 2 setes of weights, a & b and within thc call() of the layer,
and within the loss/training loop!
"""
