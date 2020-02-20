from project_config import *
FIXED_WAVELENGTH = FIXED_WAVELENGTH
FIXED_LAYERS_DISTANCE = FIXED_LAYERS_DISTANCE
def my_fftfreq(n,d=1):

    val = tf.cast(1.0 / (n*d),tf.complex128)
    result = [i for i in range(0,n)]
    N = (n-1)//2 + 1

    intervale1 = [i for i in range(0,N)]
    result[:N] = intervale1

    intervale2 = [i for i in range(-n//2 , 0)]
    result[N:] = intervale2
    result = tf.cast(result,tf.complex128)

    return tf.multiply(result,val)


def my_fft_prop(field,
                d = 75,
                nm = 1,
                res = FIXED_WAVELENGTH,
                method = 'helmholtz',
                ret_fft = None):

    # Note - my_fft_prop function recives a "field" - and not an fft_field.
    # Thus we need to do fft
    # FIXME: also notice the original funciton has a padding option. add it after we can make sure this function compiles with keras
    field = tf.cast(field,tf.complex128)
    fft_field = tf.cast(tf.signal.fft2d(field),tf.complex128)      # first convert to fft

    km = (2*PI*nm)/res

    kx = tf.reshape((my_fftfreq(fft_field.shape[1])*2*PI),[-1,1])
    ky = tf.reshape((my_fftfreq(fft_field.shape[2])*2*PI),[1,-1])

    root_km = km**2 - kx**2 - ky**2

    fstemp = tf.cast(tf.exp(1j * (tf.sqrt(root_km) - km ) * d),tf.complex128)
    fft_field = tf.cast(fft_field,tf.complex128)

    pre_ifft = tf.cast(tf.multiply(fstemp, fft_field),tf.complex128)
    result = tf.cast(tf.signal.ifft2d(pre_ifft),tf.complex128)
    return result
