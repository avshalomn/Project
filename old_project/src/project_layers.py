from project_config import *
import project_utils.prop

class phase_layer(tf.keras.layers.Layer):
    def __init__(self,shape,units = None):
        super(phase_layer,self).__init__()
        self.shape = shape
        self.units = self.shape[0] * self.shape[1]
        self.phase_weights = tf.Variable(initial_value = self.make_random_phase_wieghts(),
                                         trainable = True,
                                         dtype = tf.float32)

    def call(self,inputs):
        js = tf.constant(np.array([[0 + 1j for i in range(self.shape[1])]
                                   for k in range(self.shape[0])]))
        phase_matrix = tf.exp(tf.cast(self.phase_weights,tf.complex128) * js)
        new_var = tf.cast(inputs,'complex128') * phase_matrix
        return new_var


    def make_random_phase_wieghts(self):
        return tf.keras.backend.random_uniform_variable(shape = (self.shape[0],self.shape[1]),
                                                        low = 0,
                                                        high = 2*np.pi)



    def get_config(self):
        return super().get_config()


prop_layer = tf.keras.layers.Lambda(project_utils.prop.my_fft_prop, trainable = False)



