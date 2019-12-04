import tensorflow as tf

class layer2D(tf.keras.layers.Layer):
  def __init__(self,shape,units = None):
    super(two2layer,self).__init__()
    self.shape = shape
    self.units = shape[0] * shape[1]
    
  def get_config():
    return super().get_config()
    
  
 
