import os
import sys
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Bidirectional, Input, Reshape
from keras.layers import Lambda
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import json
working_folder = sys.argv[1]
nb_epochs = int(sys.argv[2])
gen_hidden_unit_num = int(sys.argv[3])
gen_layer_number = int(sys.argv[4])
disc_hidden_unit_num = int(sys.argv[5])
disc_layer_number = int(sys.argv[6])
classifier_hidden_unit_num = int(sys.argv[7])
classifier_layer_number = int(sys.argv[8])
print("gen_hidden_unit : ",gen_hidden_unit_num)
print("gen_layer_number : ",gen_layer_number)
print("disc_hidden_unit : ",disc_hidden_unit_num)
print("disc_layer_number : ",disc_layer_number)
print("classifier_hidden_unit : ",classifier_hidden_unit_num)
print("classifier_layer_number : ",classifier_layer_number)

opt = Adam()
batch_size = 1000
alpha_cla_adv = 0.01
alpha = 0.5
apply_epoch = 10
my_concat_1 = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))
my_concat_2 = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))
my_reshape = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1170)))
my_reshape_10 = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 10)))
my_reshape_3 = Lambda(lambda x: K.reshape(x, (1, 1, 1180)))


X = np.load('Input/inputX.npy')
Y = np.load('Label/inputy.npy')
data_train_valid, data_test, label_train_valid, label_test = train_test_split(X, Y, test_size=0.2, random_state=42)
data_train, data_valid, label_train, label_valid = train_test_split(data_train_valid, label_train_valid, test_size=0.3,
                                                                    random_state=42)

gen_input_shape = (data_train.shape[1], data_train.shape[2] + data_train.shape[2])
print('Build LSTM RNN model ...')
gen = Sequential()
for i in range(gen_layer_number):
	if (i == 0 and gen_layer_number > 1):
		gen.add(Bidirectional(LSTM(units=gen_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True), input_shape=gen_input_shape))
	elif (i == 0 and gen_layer_number <= 1):
		gen.add(Bidirectional(LSTM(units=gen_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35), input_shape=gen_input_shape))
	elif (i == gen_layer_number-1):
		gen.add(Bidirectional(LSTM(units=gen_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)))
	else:
		gen.add(Bidirectional(LSTM(units=gen_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)))
gen.add(Dense(units=10, activation='sigmoid'))

disc_input_shape = (1, 1180)
disc = Sequential()
for i in range(disc_layer_number):
	if (i == 0 and disc_layer_number > 1):
		disc.add(Bidirectional(LSTM(units=disc_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True), input_shape=disc_input_shape))
	elif (i == 0 and disc_layer_number <= 1):
		disc.add(Bidirectional(LSTM(units=disc_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35), input_shape=disc_input_shape))
	elif (i == disc_layer_number-1):
		disc.add(Bidirectional(LSTM(units=disc_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)))
	else:
		disc.add(Bidirectional(LSTM(units=disc_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)))
disc.add(Dense(units=1, activation='sigmoid'))

classifier_input_shape = (1, 10)
classifier = Sequential()
for i in range(classifier_layer_number):
	if (i == 0 and classifier_layer_number > 1):
		classifier.add(Bidirectional(LSTM(units=classifier_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True), input_shape=classifier_input_shape))
	elif (i == 0 and classifier_layer_number <= 1):
		classifier.add(Bidirectional(LSTM(units=classifier_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35), input_shape=classifier_input_shape))
	elif (i == classifier_layer_number-1):
		classifier.add(Bidirectional(LSTM(units=classifier_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)))
	else:
		classifier.add(Bidirectional(LSTM(units=classifier_hidden_unit_num, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)))
classifier.add(Dense(units=1170, activation='elu'))
classifier.add(Reshape((30, 39)))


# Build combined model gen_disc
disc.trainable = False
z = Input(shape=(data_train.shape[1], data_train.shape[2]))
y = Input(shape=(data_train.shape[1], data_train.shape[2]))
combined = my_concat_2([z, y])
generated_label = my_reshape_10(gen(combined))
flatten_y = my_reshape(y)
generated_label_speech = my_concat_1([generated_label, flatten_y])
disc_fake = disc(generated_label_speech)
gen_disc = Model([z, y], disc_fake)


# Build combined model classifier_disc
disc.trainable = False
x = Input(shape=(1, 10))
generated_speech_ = my_reshape(classifier(x))
generated_label_speech_ = my_concat_1([my_reshape_10(x), generated_speech_])
disc_unlabel = disc(generated_label_speech_)
cla_disc = Model(x, disc_unlabel)


# Build combined model classifier_gen_disc
disc.trainable = False
gen.trainable = False
disc_cla = cla_disc(generated_label)
cla_gen_disc = Model([z, y], disc_cla)


print("Compiling ...")
gen.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
gen.summary()
disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
disc.summary()
classifier.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
classifier.summary()
gen_disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
cla_disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
cla_gen_disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Training ...")
for epoch in range(nb_epochs):
        if epoch >= apply_epoch:
            alpha_p = 0.1
        else:
            alpha_p = 0.0
        # Train Discriminator
        noise = np.random.normal(0, 1, (batch_size, data_train.shape[1], data_train.shape[2]))
        idx = np.random.randint(0, data_train.shape[0], batch_size)
        real_data = data_train[idx]
        noise_real_data = np.concatenate((noise, real_data), axis=2)
        real_data_reshaped = np.reshape(real_data, (batch_size, -1))
        idx_valid = np.random.randint(0, data_valid.shape[0], batch_size)
        unlabel = np.reshape(label_valid[idx_valid], (batch_size, 1, 10))
        unlabel_data = data_valid[idx_valid]
        real_label = np.reshape(label_train[idx], (batch_size, 1, 10))

        gen_label = gen.predict(noise_real_data)
        fake_ = np.reshape(np.concatenate((gen_label, real_data_reshaped), axis=1), (batch_size, 1, 1180))
        real_ = np.concatenate((real_label, np.reshape(real_data, (batch_size, 1, 1170))), axis=2)

        d_loss_fake = disc.train_on_batch(fake_, np.zeros((batch_size, 1)))
        d_loss_real = disc.train_on_batch(real_, np.ones((batch_size, 1)))
        d_loss_cla = cla_disc.train_on_batch(unlabel, np.zeros((batch_size, 1)))
        d_loss = np.add(np.add(d_loss_real, d_loss_fake), d_loss_cla) / 3

        # Train Generator
        valid_y = np.reshape(np.array([1] * batch_size), (batch_size, 1))

        g_loss = gen_disc.train_on_batch([noise, real_data], valid_y)

        # Train Classifier
        R_L = classifier.train_on_batch(real_label, real_data)
        R_P = classifier.train_on_batch(np.reshape(gen_label, (batch_size, 1, 10)), real_data)
        c_loss_dis = cla_disc.train_on_batch(unlabel, valid_y)
        c_loss = np.add(np.add(R_L, np.multiply(alpha_p, R_P)), np.multiply(alpha_cla_adv,
                                                                            np.multiply(alpha, c_loss_dis)))
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f], [C loss: %f]" %
               (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], c_loss[0]))

# file_gen_history = open(working_folder+"/gen_history.json","w")
# file_gen_history.write(json.dumps(gen_history.history))
# file_gen_history.close()
# file_disc_history = open(working_folder+"/disc_history.json","w")
# file_disc_history.write(json.dumps(disc_history.history))
# file_disc_history.close()
# file_classifier_history = open(working_folder+"/classifier_history.json","w")
# file_classifier_history.write(json.dumps(classifier_history.history))
# file_classifier_history.close()

gen.save(working_folder+'/gen_model.h5')
disc.save(working_folder+'/disc_model.h5')
classifier.save(working_folder+'/classifier_model.h5')
