from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA

# Good references that helped me along the way:
# [1] http://home.mit.bme.hu/~hadhazi/Oktatas/NN18/dem3/html_demo/CIFAR-10Demo.html
# [2] http://neuralnetworksanddeeplearning.com/chap1.html


def config_env():

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons

    # tf Graph input - Notice we do not specify the number of inputs since the batch size may change
    x = tf.placeholder(tf.float32, [None, n_input])
    # Y = tf.placeholder(tf.int64, [None, n_classes])
    y = tf.placeholder(tf.int64, [None])

    # Store layers weight & bias
    w = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    b = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    return x, y, w, b


# Create model
def multilayer_perceptron(X, weights, biases):
    # This function returns the mathematical formulation embedded within the neural network,
    # where the weights and biases of all the layers are multiplied by the intermediate values
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.identity(out_layer, name="op_to_restore")
    return out_layer


def train():
    # Load DATA: x_train: (num_samples, 32, 32, 3)  ,  y_train: (num_samples,) numbers in [0-9]
    (x_train, y_train), _ = cifar10.load_data()
    if verbosity:
        print("Shape of the X-TRAIN: {0}".format(x_train.shape))
        print("Shape of the Y-TRAIN: {0}".format(y_train.shape))
    # Reshape our data. Our images have the size (50000, 32, 32, 3), where 50000 images are represented by 32x32 pixels,
    # each of them specified in RGB form, hence 3 values. We should reshape them as: 32x32x3=3072
    x_train = np.reshape(x_train, (-1, 3072))
    y_train = np.squeeze(y_train)
    x_train /= 255  # Normalization of pixel values (to [0-1] range)
    # Training cycle
    batch_size = 128
    training_epochs = 10*(x_train.shape[0] / batch_size)
    display_step = 100  # Display results form a limited number of epochs
    # Define loss and optimizer functions (Core of the training process)
    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y, n_classes), logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(training_epochs):
        s = np.arange(x_train.shape[0])
        np.random.shuffle(s)
        x_tr = x_train[s]
        y_tr = y_train[s]
        batch_x = x_tr[:batch_size]
        batch_y = y_tr[:batch_size]
        # Run optimization op (backprop) and cost op (to get loss value). The first argument contains the actions to
        # perform. In our case, train_op reflects the intention to train the model and update the weights whereas
        # loss_op just requires to compute the losses from the previously defined loss function
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        # Average loss per image used to train
        loss = loss / batch_size
        # Display logs per epoch step
        if epoch % display_step == 0:
            # print("  Epoch:", '%05d' % (epoch + 1), "cost={:.9f}".format(loss), " training images: {0}".format(s[:batch_size]))
            print("  Epoch:", '%05d' % (epoch + 1), "cost={:.9f}".format(loss))
    print("Optimization Finished!")
    # Once done, we save the computed model. `save` method will call `export_meta_graph` implicitly.
    # The model (graph) will be stored at files:my-model.meta
    # Create a saver object which will save all the variables
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(os.getcwd(), 'CBG_model_HW1'))


def debugging():
    # Test model
    # Load DATA: x_test: (num_samples, 32, 32, 3)  ,  y_test: (num_samples,) numbers in [0-9]
    _, (x_test, y_test) = cifar10.load_data()
    x_test = np.reshape(x_test, (-1, 3072))
    y_test = np.squeeze(y_test)
    x_test /= 255  # Normalization of pixel values (to [0-1] range)
    if verbosity:
        print("Shape of the X-TEST: {0}".format(x_test.shape))
        print("Shape of the Y-TEST: {0}".format(y_test.shape))
    # Restore the precomputed model
    new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'CBG_model_HW1.meta'))
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')
    print(all_vars)
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)
    # Re-initialize parameters
    graph = tf.get_default_graph()
    restored_logits = graph.get_tensor_by_name("op_to_restore:0")
    # Compute accuracy for test Images
    pred = tf.nn.softmax(restored_logits)  # Apply softmax to logits
    # Define the correct prediction as the maximum number to select amongst the output (last layer)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_test[:1], Y: y_test[:1]}))


def test(x_input):
    x_input = np.reshape(x_input, (-1, 3072))
    x_input /= 255  # Normalization of pixel values (to [0-1] range)
    # Restore the precomputed model
    new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'CBG_model_HW1.meta'))
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
    # Re-initialize parameters
    graph = tf.get_default_graph()
    restored_logits = graph.get_tensor_by_name("op_to_restore:0")
    # Compute accuracy for test Images
    pred = tf.nn.softmax(restored_logits)  # Apply softmax to logits
    # Define the correct prediction as the maximum number to select amongst the output (last layer)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


if __name__ == '__main__':
    # Check required input arguments
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print("Error - Please input the correct number of arguments")
        sys.exit()  # Terminate execution - chico malo

    # Initialization

    # Parameters
    learning_rate = 0.0001  # Learning rate for optimization algorithm (GD, Adam, etc)
    n_input = 3072  # CIFAR data input with images 32x32x3
    n_classes = 10  # CIFAR total classes (0-9 classes)
    verbosity = False

    # Configure the environment
    [X, Y, weights, biases] = config_env()
    print(X.shape)

    # Construct model
    logits = multilayer_perceptron(X, weights, biases)

    # Instantiate an Object Session
    sess = tf.Session()

    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        # Train the model
        train()
    if len(sys.argv) == 3 and sys.argv[1] == 'test':
        # Preprocessing of image
        x_input = sys.argv[2]
        x_input = np.reshape(x_input, (-1, 3072))  # Reshape input
        x_input /= 255  # Normalization of pixel values (to [0-1] range)
        # Just one image
        test(x_input)
    if len(sys.argv) == 2 and sys.argv[1] == 'debugging':
        # Train the model
        train()
        _, (x_test, y_test) = cifar10.load_data()
        x_test = np.reshape(x_test, (-1, 3072))
        y_test = np.squeeze(y_test)
        x_test /= 255  # Normalization of pixel values (to [0-1] range)
        # Just one image
        x_test = x_test[:1]
        y_test = y_test[:1]
        debugging(x_test)