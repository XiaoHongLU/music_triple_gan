import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Bidirectional
from keras.optimizers import Adam
from GenreFeatureData import GenreFeatureData  # local python class with Audio feature extraction (librosa)
from sklearn.model_selection import train_test_split
import numpy as np
# Turn off TF verbose logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# genre_features = GenreFeatureData()
# genre_features.load_preprocess_data()
# genre_features.load_deserialize_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

batch_size = 1000
nb_epochs = 50
X = np.load('Input/inputX.npy')
Y = np.load('Label/inputy.npy')
data_train_valid, data_test, label_train_valid, label_test = train_test_split(X, Y, test_size=0.2, random_state=42)
data_train, data_valid, label_train, label_valid = train_test_split(data_train_valid, label_train_valid, test_size=0.3,
                                                                    random_state=42)

#print("Training X shape: " + str(genre_features.train_X.shape))
#print("Training Y shape: " + str(genre_features.train_Y.shape))
#print("Dev X shape: " + str(genre_features.dev_X.shape))
#print("Dev Y shape: " + str(genre_features.dev_Y.shape))
#print("Test X shape: " + str(genre_features.test_X.shape))
#print("Test Y shape: " + str(genre_features.test_X.shape))

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
model.fit(data_train, label_train, batch_size=batch_size, epochs=nb_epochs)

print("\nValidating ...")
score, accuracy = model.evaluate(data_valid,label_valid, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)
model.save('/home/s1679450/2l.h5')

#print("\nTesting ...")
#score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1)
#print("Test loss:  ", score)
#print("Test accuracy:  ", accuracy)
