from project_config import *
import project_layers
import project_loss_funcs
import project_utils.utils
def make_square_target(shape,N):
    h,w = shape[0],shape[1]
    out = np.zeros((h,w)) + 0.000000001j
    out[h//4:(h//5*3) , h//4:(h//5)*3] = 1 + 0.000000001j
    return np.array([out for i in range(N)])
def make_target():
    out =  np.array([np.arange(0,10)/10 for i in range(0,10)])
    out = out + 0.000001j
    out[0:5,:] = 0 + 0.0001j
    out[6:10,:] = 0 + 0.000001j
    return out
def make_inputs():
    out =  np.random.rand(N,20,20)
    out = out + 0.00001j
    out[:,0:3,:] = 0 + 0.00000001j
    out[:,6:9,:] = 0 + 0.00000001j
    return out





#######################
### CONSTANTS #########
#######################
N = 10000
inputs = idx2numpy.convert_from_file(MNIST_DIGIT_TRAIN_IMGS_PATH) / 255
targets = idx2numpy.convert_from_file(MNIST_DIGIT_TRAIN_LABELS_CONVERTED_PATH)
targets = np.complex128(targets) + 0.00000000001j
inputs = np.complex128(inputs) + 0.00000000001j
SHAPE = (inputs.shape[1],inputs.shape[2])
EPOCHS = 3
BATCH_SIZE = 4
N_STEPS =len(inputs)//BATCH_SIZE
OPTIMIZER = tf.keras.optimizers.SGD(lr = 0.01)
loss_fn = project_loss_funcs.only_phase_loss
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]


my_dense0 = project_layers.phase_layer(shape=SHAPE)
my_dense1 = project_layers.phase_layer(shape=SHAPE)
my_dense2 = project_layers.phase_layer(shape=SHAPE)
my_dense3 = project_layers.phase_layer(shape=SHAPE)
my_dense4 = project_layers.phase_layer(shape=SHAPE)
prop_layer0 = project_layers.prop_layer
prop_layer1 = project_layers.prop_layer
prop_layer2 = project_layers.prop_layer
prop_layer3 = project_layers.prop_layer
prop_layer4 = project_layers.prop_layer
##############
### model ####
##############
model = tf.keras.models.Sequential()
model.add(prop_layer0)
model.add(my_dense0)
model.add(prop_layer1)
model.add(my_dense1)
model.add(prop_layer2)
model.add(my_dense2)
model.add(prop_layer3)
model.add(my_dense3)
model.add(prop_layer4)
model.add(my_dense4)
# model.compile(optimizer = 'adagrad',loss = 'mean_squared_error')

TRAIN = 1
if TRAIN:
    for epoch in range(1,EPOCHS+1):
        print("Epoch {}/{}".format(epoch, EPOCHS))
        for step in range(1,N_STEPS+1):
            X_BATCH,y_batch = project_utils.utils.random_batch(X = inputs, y = targets,
                                                          batch_size=BATCH_SIZE)
            X_BATCH = tf.cast(X_BATCH,tf.complex128)
            y_batch = tf.cast(y_batch,tf.complex128)
            with tf.GradientTape() as tape:
                y_pred = model(X_BATCH,training = True)
                l = loss_fn(y_pred,y_batch)
                main_loss = tf.keras.backend.mean(l,axis = [0],keepdims =
                False)
                loss = tf.math.add_n([main_loss] + model.losses)
            ## FIXME: MAKE MY OWN GRADIENT AND TAPE!
            gradients = tape.gradient(loss,model.trainable_variables)
            # print("\n*** pre-gradients weights[0][0] ***\n",model.weights[0][0])
            OPTIMIZER.apply_gradients(zip(gradients, model.trainable_variables))
            # print("\n*** post-gradients weights[0][0] ***\n",model.weights[0][0])
            mean_loss(loss)

            for metric in metrics:
                metric(y_batch, y_pred)
            project_utils.utils.print_status_bar(step * BATCH_SIZE, len(targets), mean_loss, metrics)
        project_utils.utils.print_status_bar(len(targets), len(targets), mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()


PREDICT = 1
if PREDICT:
    x = inputs[0].reshape(1,SHAPE[0],SHAPE[1])
    y = model.predict(x)
    print(x[0][0])
    print(y[0][0])

    plt.subplot(311)
    plt.imshow(x[0].real,'gist_heat')
    plt.subplot(312)
    plt.imshow(targets[0].real,'gist_heat')
    plt.subplot(313)
    plt.imshow((y[0].real)**2,'gist_heat')
    plt.show()


