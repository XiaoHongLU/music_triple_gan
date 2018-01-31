import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.model_selection import train_test_split
import os

INPUT_DIMENSION = 39
HIDDEN_DIMENSION = 200
OUTPUT_DIMENSION = 10
LEARNING_RATE = 0.03
FRAME_WINDOW = 30
EPOCH = 10000
DISPLAY_EPOCH = 100


def create_variables():
    w1_lstm = tf.Variable(tf.random_normal([2*HIDDEN_DIMENSION, OUTPUT_DIMENSION]))
    b1_lstm = tf.Variable(tf.random_normal([OUTPUT_DIMENSION]))

    variables = {
        'w1_lstm': w1_lstm, 'b1_lstm': b1_lstm
    }

    return variables


# Load data
X = np.load('Input/inputX.npy')
Y = np.load('Label/inputy.npy')

data_train_valid, data_test, label_train_valid, label_test = train_test_split(X, Y, test_size=0.2, random_state=42)
data_train, data_valid, label_train, label_valid = train_test_split(data_train_valid, label_train_valid, test_size=0.3,
                                                                    random_state=42)


def model(inputs, variables, layername):

    # Define lstm cells with tensorflow
    inputs = tf.unstack(inputs, FRAME_WINDOW, 1)

    with tf.variable_scope(layername) as scope:
    # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0)
    # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0)

        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, scope=scope)

    return tf.nn.sigmoid(tf.matmul(outputs[-1], variables['w1_lstm']) + variables['b1_lstm'])


graph = tf.Graph()
with graph.as_default():
    tf_train_data = tf.placeholder(tf.float32, shape=[None, FRAME_WINDOW, INPUT_DIMENSION])
    tf_train_label = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIMENSION])

    variables = create_variables()

    model_ = model
    logits = model_(tf_train_data, variables, 'layer1')

    loss = tf.losses.sigmoid_cross_entropy(tf_train_label, logits)
    acc = tf.equal(logits, tf_train_label)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

if not os.path.isfile(os.path.join(os.getcwd(),"Model/RNN_200/RNN_200.ckpt.meta")):
    with tf.Session(graph=graph) as session:
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        progress = open(os.path.join(os.getcwd(), 'Model/RNN_200/RNN_200.txt'), 'wb')
        for step in range(EPOCH):
            feed_dict = {tf_train_data: data_train, tf_train_label: label_train}
            _, l, acc_train = session.run([optimizer, loss, acc], feed_dict=feed_dict)

            if step % DISPLAY_EPOCH == 0:
                loss_valid, predict_valid, acc_valid = session.run([loss, logits, acc], feed_dict={tf_train_data: data_valid,
                                                                                   tf_train_label: label_valid})
                print (acc_train[0])
                #cc = np.corrcoef(predict_valid, label_valid)
                print ("step %d, train : loss is %g" % (step, l))
                print ("step %d, validation : loss is %g"%(step, loss_valid))
                progress.write("step %d, train : loss is %g" % (step, l))
                progress.write('\n')
                progress.write("step %d, validation : loss is %g" % (step, loss_valid))
                progress.write('\n')

        loss_test, predict_test = session.run([loss, logits],
                                         feed_dict={tf_train_data: data_test,
                                                    tf_train_label: label_test})
        #cc = np.corrcoef(label_test, predict_test)
        print("test dataset : loss is %g" % loss_test)
        progress.write("test dataset : loss is %g" % loss_test)
        progress.write('\n')

        saver = tf.train.Saver()
        saver.save(session, os.path.join(os.getcwd(), "Model/RNN_200/RNN_200.ckpt"))
else:
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        saver.restore(session, os.path.join(os.getcwd(), 'Model/RNN_200/RNN_200.ckpt'))

        predict = session.run([logits], feed_dict={tf_train_data: data_test, tf_train_label: label_test})
