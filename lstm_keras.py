import os
import sys
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import json
working_folder = sys.argv[1]
opt = Adam()
batch_size = 1000
nb_epochs = 300
X = np.load('Input/inputX.npy')
Y = np.load('Label/inputy.npy')
data_train_valid, data_test, label_train_valid, label_test = train_test_split(X, Y, test_size=0.2, random_state=42)
data_train, data_valid, label_train, label_valid = train_test_split(data_train_valid, label_train_valid, test_size=0.3,
                                                                    random_state=42)

input_shape = (data_train.shape[1], data_train.shape[2])
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=100, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))

model.add(Bidirectional(LSTM(units=100, dropout=0.05, recurrent_dropout=0.35, return_sequences=False)))
model.add(Dense(units=10, activation='sigmoid'))

print("Compiling ...")
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

print("Training ...")
history = model.fit(data_train, label_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(data_valid,label_valid))
file_history = open(working_folder+"/history.json","w")
file_history.write(json.dumps(history.history))
file_history.close()
model.save(working_folder+'/model.h5')
