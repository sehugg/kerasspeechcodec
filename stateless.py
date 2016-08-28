#!/usr/bin/python

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, MaxoutDense, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianNoise
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.layers.advanced_activations import *
import numpy as np
import random
import sys
import c2

cmpsperword = c2.COMPONENTS_PER_WORD
maxlen = 50
nbatch = 64

# read input
filename = sys.argv[1]
print ("Reading %s..." % filename)
with open(filename,'rb') as f:
  input_frames = c2.decode_c2file(f)

# split frames into phrases at silence gaps
print ("Read %d frames" % len(input_frames))
input_phrases = c2.split_phrases(input_frames, -0.8, 10, 10)
print ("Detected %d phrases" % len(input_phrases))

# concatenate phrases back into frames
input_words = []
for ph in input_phrases:
    input_words.extend(ph)
print ("Using %d frames" % len(input_words))

# build sequences

redun_step = int(len(input_words) / 500000) + 1

sequences = []
next_words = []
for i in range(0, len(input_words) - maxlen, redun_step):
  sequences.append(input_words[i: i + maxlen])
  next_words.append(input_words[i + maxlen])

X = np.zeros((len(sequences), maxlen, cmpsperword), dtype=np.float32)
Y = np.zeros((len(sequences), cmpsperword), dtype=np.float32)

for i, seq in enumerate(sequences):
  for t, w in enumerate(seq):
    X[i, t] = w
  Y[i] = next_words[i]

print("Created",len(sequences),"sequences")

# build the model
nunits = 512
dropout = 0.5

print('Build model...')
model = Sequential()
model.add(LSTM(nunits, consume_less='gpu', return_sequences=True, input_shape=(maxlen, cmpsperword)))
model.add(Dropout(dropout))
model.add(LSTM(nunits, consume_less='gpu', return_sequences=True, go_backwards=True))
model.add(Dropout(dropout))
model.add(LSTM(nunits, consume_less='gpu', return_sequences=False, go_backwards=True))
model.add(Dropout(dropout))
model.add(Dense(nunits*2, activation='relu'))
model.add(Dense(cmpsperword, activation='tanh'))

model.compile(loss='mae', optimizer='adadelta')
model.summary()

# read old weights, if present
weightsfn = 'init_weights.h5'
try:
  model.load_weights(weightsfn)
  print("loaded",weightsfn)
except:
  print('could not load',weightsfn)

# train the model, output generated voice after each iteration

numseq = 10000
numinc = 5000

for iteration in range(1, 100):

    numseq = min(numseq+numinc, len(X))

    print()
    print('-' * 50)
    print('Iteration', iteration, 'num=',numseq)
    XX = X[0:numseq]
    YY = Y[0:numseq]

    model.fit(XX, YY, batch_size=nbatch, nb_epoch=10, shuffle=True)

    # start with initial random seed
    start_index = random.randint(0, min(len(input_words) - maxlen - 1, numseq*redun_step))
    generated = []
    sentence = input_words[start_index: start_index + maxlen]

    # generate speech
    for i in range(1000):
        x = np.zeros((1, maxlen, cmpsperword), dtype=np.float)
        for t, w in enumerate(sentence):
            x[0, t] = w

        preds = model.predict(x, verbose=0)[0]
        next_word = preds

        generated.append(next_word)
        sentence = sentence[1:]
        sentence.append(next_word)

    # output generated speech
    outfn = 'output_%d.c232' % iteration
    print("Writing ",outfn)
    with open(outfn, 'wb') as f:
        c2.encode_c2file(f, generated)
    
    print(np.mean(generated), np.std(generated))

    # save weights for this iteration
    weightsfn = 'weights_%d.h5' % iteration
    model.save_weights(weightsfn, overwrite=True)

