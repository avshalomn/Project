import tensorflow as tf

@tf.function
def only_phase_loss(y_pred,y_true):
    get_real = tf.math.real
    get_imag = tf.math.imag

    pred_real = get_real(y_pred)
    pred_imag = get_imag(y_pred)
    true_real = get_real(y_true)
    true_imag = get_imag(y_true)

    pred_phase = tf.math.atan2(pred_imag,pred_real)
    true_phase = tf.math.atan2(true_imag,true_real)

    loss = 0.5 * (tf.math.abs(true_phase - pred_phase))**2

    return loss
