import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
import os

INPUT_DIMENSION = 39
SONG_DIMENSION = 1170
HIDDEN_DIMENSION = 100
OUTPUT_DIMENSION = 10
DISC_OUTPUT_DIMENSION = 1
LEARNING_RATE = 0.03
FRAME_WINDOW = 30
EPOCH = 1000
DISPLAY_EPOCH = 10


def create_variables():
    gen_w1_lstm = tf.Variable(tf.random_normal([2 * HIDDEN_DIMENSION, OUTPUT_DIMENSION]))
    gen_b1_lstm = tf.Variable(tf.random_normal([OUTPUT_DIMENSION]))

    disc_w1_lstm = tf.Variable(tf.random_normal([2 * HIDDEN_DIMENSION, DISC_OUTPUT_DIMENSION]))
    disc_b1_lstm = tf.Variable(tf.random_normal([DISC_OUTPUT_DIMENSION]))

    cls_w1_lstm = tf.Variable(tf.random_normal([2 * HIDDEN_DIMENSION, SONG_DIMENSION]))
    cls_b1_lstm = tf.Variable(tf.random_normal([SONG_DIMENSION]))

    variables = {
        'gen_w1_lstm': gen_w1_lstm, 'gen_b1_lstm': gen_b1_lstm,
        'disc_w1_lstm': disc_w1_lstm, 'disc_b1_lstm': disc_b1_lstm,
        'cls_w1_lstm': cls_w1_lstm, 'cls_b1_lstm': cls_b1_lstm

    }

    return variables


# Load data
X = np.load('Input/inputX.npy')
Y = np.load('Label/inputy.npy')

data_train_valid, data_test, label_train_valid, label_test = train_test_split(X, Y, test_size=0.2, random_state=42)
data_train, data_valid, label_train, label_valid = train_test_split(data_train_valid,
                                                                    label_train_valid,
                                                                    test_size=0.3,
                                                                    random_state=42)

train_batch_size = np.shape(data_train)[0]
valid_batch_size = np.shape(data_valid)[0]
test_batch_size = np.shape(data_test)[0]
flatten_data_train = np.reshape(data_train, (train_batch_size, 1170))
flatten_data_valid = np.reshape(data_valid, (valid_batch_size, 1170))
flatten_data_test = np.reshape(data_test, (test_batch_size, 1170))


# Generator
def generator(inputs, conditional, variables, reuse=False):
    # Define lstm cells with tensorflow
    inputs = tf.concat(values=[inputs, conditional], axis=2)
    inputs = tf.unstack(inputs, FRAME_WINDOW, 1)

    with tf.variable_scope('Generator', reuse=reuse):
        with tf.variable_scope('gen_layer1') as scope:
            # Forward direction cell
            lstm_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)
            # Backward direction cell
            lstm_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            rnn1_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32,
                                                              scope=scope)

        with tf.variable_scope('gen_layer2') as scope1:
            # Forward direction cell
            lstm2_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            # Backward direction cell
            lstm2_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            rnn2_outputs, _, _ = rnn.static_bidirectional_rnn(lstm2_fw_cell, lstm2_bw_cell, rnn1_outputs,
                                                              dtype=tf.float32,
                                                              scope=scope1)

        with tf.variable_scope('gen_layer3') as scope2:
            # Forward direction cell
            lstm3_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            # Backward direction cell
            lstm3_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            rnn3_outputs, _, _ = rnn.static_bidirectional_rnn(lstm3_fw_cell, lstm3_bw_cell, rnn2_outputs,
                                                              dtype=tf.float32,
                                                              scope=scope2)

    return tf.nn.sigmoid(tf.matmul(rnn3_outputs[-1], variables['gen_w1_lstm']) + variables['gen_b1_lstm'])


# Discriminator
def discriminator(inputs, conditional, variables, reuse=False):
    # Define lstm cells with tensorflow
    shape = tf.shape(inputs)
    inputs = tf.concat(values=[inputs, conditional], axis=1)
    inputs = tf.reshape(inputs, [shape[0], 1, OUTPUT_DIMENSION + SONG_DIMENSION])
    inputs = tf.unstack(inputs, 1, 1)

    with tf.variable_scope('Discriminator', reuse=reuse):
        with tf.variable_scope('disc_layer1') as scope:
            # Forward direction cell
            lstm_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)
            # Backward direction cell
            lstm_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            rnn1_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32,
                                                              scope=scope)

        # with tf.variable_scope('disc_layer2') as scope1:
        #     # Forward direction cell
        #     lstm2_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
        #                                                  input_keep_prob=0.05, state_keep_prob=0.35)
        #
        #     # Backward direction cell
        #     lstm2_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
        #                                                  input_keep_prob=0.05, state_keep_prob=0.35)
        #
        #     rnn2_outputs, _, _ = rnn.static_bidirectional_rnn(lstm2_fw_cell, lstm2_bw_cell, rnn1_outputs,
        #                                                       dtype=tf.float32,
        #                                                       scope=scope1)

    return tf.nn.sigmoid(tf.matmul(rnn1_outputs[-1], variables['disc_w1_lstm']) + variables['disc_b1_lstm'])


# Classifier
def classifier(inputs, variables, reuse=False):
    # Define lstm cells with tensorflow
    shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [shape[0], 1, OUTPUT_DIMENSION])
    inputs = tf.unstack(inputs, 1, 1)

    with tf.variable_scope('Classifier', reuse=reuse):
        with tf.variable_scope('cls_layer1') as scope:
            # Forward direction cell
            lstm_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)
            # Backward direction cell
            lstm_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
                                                         input_keep_prob=0.05, state_keep_prob=0.35)

            rnn1_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32,
                                                              scope=scope)

        # with tf.variable_scope('cls_layer2') as scope1:
        #     # Forward direction cell
        #     lstm2_fw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
        #                                                  input_keep_prob=0.05, state_keep_prob=0.35)

        #     # Backward direction cell
        #     lstm2_bw_cell = rnn.DropoutWrapper(rnn.BasicLSTMCell(HIDDEN_DIMENSION, forget_bias=1.0),
        #                                                  input_keep_prob=0.05, state_keep_prob=0.35)

        #     rnn2_outputs, _, _ = rnn.static_bidirectional_rnn(lstm2_fw_cell, lstm2_bw_cell, rnn1_outputs,
        #                                                       dtype=tf.float32,
        #                                                       scope=scope1)

    return tf.matmul(rnn1_outputs[-1], variables['cls_w1_lstm']) + variables['cls_b1_lstm']


graph = tf.Graph()
with graph.as_default():
    tf_noise = tf.placeholder(tf.float32, shape=[None, FRAME_WINDOW, INPUT_DIMENSION])
    tf_train_data = tf.placeholder(tf.float32, shape=[None, FRAME_WINDOW, INPUT_DIMENSION])
    tf_train_label = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIMENSION])
    tf_train_data_ = tf.placeholder(tf.float32, shape=[None, SONG_DIMENSION])
    tf_unlabel_data = tf.placeholder(tf.float32, shape=[None, FRAME_WINDOW, INPUT_DIMENSION])
    tf_unlabel_label = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIMENSION])
    BATCH_SIZE = tf.placeholder(tf.int32)

    variables = create_variables()

    # Generator
    gen = generator(tf_noise, tf_train_data, variables)

    # Three Classifiers
    cls_real = classifier(tf_train_label, variables)
    cls_fake = classifier(gen, variables, True)
    cls_unlabel = classifier(tf_unlabel_label, variables, True)

    # Three Discriminators
    disc_real = discriminator(tf_train_label, tf_train_data_, variables)
    disc_fake = discriminator(gen, tf_train_data_, variables, True)
    disc_unlabel = discriminator(tf_unlabel_label, cls_unlabel, variables, True)

    # loss for each network
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones([BATCH_SIZE, 1])))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros([BATCH_SIZE, 1])))
    D_loss_cls = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_unlabel, labels=tf.zeros([tf.shape(tf_unlabel_data)[0], 1])))
    disc_loss = D_loss_real + D_loss_fake + D_loss_cls

    cls_disc_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_unlabel, labels=tf.ones([tf.shape(tf_unlabel_data)[0], 1])))
    mean, var = tf.nn.moments(tf_train_data_, axes=0)
    cls_R_L = tf.losses.mean_squared_error(tf_train_data_, cls_real) / tf.reduce_mean(var)
    cls_R_P = tf.losses.mean_squared_error(tf_train_data_, cls_fake) / tf.reduce_mean(var)
    cls_loss = cls_R_L + cls_R_P + cls_disc_loss

    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones([BATCH_SIZE, 1])))

    gen_vars = []
    disc_vars = []
    cls_vars = []
    for tf_var in tf.trainable_variables():
        if 'Generator' in tf_var.name:
            gen_vars.append(tf_var)
        if 'Discriminator' in tf_var.name:
            disc_vars.append(tf_var)
        if 'Classifier' in tf_var.name:
            cls_vars.append(tf_var)
    gen_vars.extend([variables['gen_w1_lstm'], variables['gen_b1_lstm']])
    disc_vars.extend([variables['disc_w1_lstm'], variables['disc_b1_lstm']])
    cls_vars.extend([variables['cls_w1_lstm'], variables['cls_b1_lstm']])

    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
    disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
    cls_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Classifier')

    # Build Optimizers
    with tf.control_dependencies(gen_update_ops):
        optimizer_gen = tf.train.AdamOptimizer(LEARNING_RATE).minimize(gen_loss, var_list=gen_vars)
    with tf.control_dependencies(disc_update_ops):
        optimizer_disc = tf.train.AdamOptimizer(LEARNING_RATE).minimize(disc_loss, var_list=disc_vars)
    with tf.control_dependencies(cls_update_ops):
        optimizer_cls = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cls_loss, var_list=cls_vars)

if not os.path.isfile(os.path.join(os.getcwd(), "Model/Triple_GAN_RNN_G3_D1_C1/Triple_GAN_RNN_G3_D1_C1.ckpt.meta")):
    with tf.Session(graph=graph) as session:
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        progress = open(os.path.join(os.getcwd(), 'Model/Triple_GAN_RNN_G3_D1_C1/Triple_GAN_RNN_G3_D1_C1.txt'), 'w')
        for step in range(EPOCH):
            train_noise = np.random.uniform(-1, 1, size=[train_batch_size, FRAME_WINDOW, INPUT_DIMENSION])
            feed_dict = {tf_noise: train_noise, tf_train_data: data_train, tf_train_label: label_train,
                         tf_train_data_: flatten_data_train, BATCH_SIZE: train_batch_size, tf_unlabel_data: data_valid,
                         tf_unlabel_label: label_valid}
            _, disc_l, result = session.run([optimizer_disc, disc_loss, disc_real], feed_dict=feed_dict)

            _, gen_l = session.run([optimizer_gen, gen_loss], feed_dict=feed_dict)

            _, cls_l = session.run([optimizer_cls, cls_loss], feed_dict=feed_dict)

            if step % DISPLAY_EPOCH == 0:
                print ("step %d, train : gen_loss is %g, disc_loss is %g, cls_loss is %g" % (step, gen_l, disc_l, cls_l))
                progress.write("step %d, train : gen_loss is %g, disc_loss is %g,, cls_loss is %g" % (step, gen_l,
                                                                                                     disc_l, cls_l))
                progress.write('\n')

        saver = tf.train.Saver()
        saver.save(session, os.path.join(os.getcwd(), "Model/Triple_GAN_RNN_G3_D1_C1/Triple_GAN_RNN_G3_D1_C1.ckpt"))
        test_noise = np.random.uniform(-1, 1, size=[test_batch_size, FRAME_WINDOW, INPUT_DIMENSION])
        disc_loss_test, gen_loss_test, cls_loss_test = session.run([disc_loss, gen_loss, cls_loss], feed_dict={tf_noise: test_noise,
                                                                                      tf_train_data: data_test,
                                                                                      tf_train_label: label_test,
                                                                                      tf_train_data_: flatten_data_test,
                                                                                      BATCH_SIZE: test_batch_size,
                                                                                    tf_unlabel_data: data_valid,
                                                                                    tf_unlabel_label: label_valid
                                                                                                            })

        print("test dataset : gen_loss is %g, disc_loss is %g, cls_loss is %g" % (gen_loss_test, disc_loss_test, cls_loss_test))
        progress.write(
            "test dataset : gen_loss is %g, disc_loss is %g, cls_loss is %g" % (gen_loss_test, disc_loss_test, cls_loss_test))
        progress.write('\n')
        progress.close()
else:
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        saver.restore(session, os.path.join(os.getcwd(), 'Model/Triple_GAN_RNN_G3_D1_C1/Triple_GAN_RNN_G3_D1_C1.ckpt'))

        train_noise = np.random.uniform(-1., 1., size=[train_batch_size, FRAME_WINDOW, INPUT_DIMENSION])
        test_noise = np.random.uniform(-1., 1., size=[test_batch_size, FRAME_WINDOW, INPUT_DIMENSION])

        train_gen = np.asarray(session.run(gen, feed_dict={tf_noise: train_noise, tf_train_data: data_train,
                                                           tf_train_label: label_train,
                                                           tf_train_data_: flatten_data_train,
                                                           BATCH_SIZE: train_batch_size}))

        test_gen = np.asarray(session.run(gen, feed_dict={tf_noise: test_noise, tf_train_data: data_test,
                                                          tf_train_label: label_test,
                                                          tf_train_data_: flatten_data_test,
                                                          BATCH_SIZE: test_batch_size}))

        print ("Train Hamming Loss is %g" % (hamming_loss(label_train, np.round(train_gen))))
        print ("Test Hamming Loss is %g" % (hamming_loss(label_test, np.round(test_gen))))
