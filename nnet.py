import numpy as np
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import argparse
import sys

file_name = 'generated.txt'
window_size = 24 #default window size

class MyParser(argparse.ArgumentParser):
	def error(self,message):
		sys.stderr.write('Error: %s\n' %message)
		self.print_help()
		sys.exit()


def load_dataset(file_name):

        pattern_list = []
        f1 = open(file_name,'r')
        line = f1.readlines()
        for element in line:
                element = element[9:]
                #element = element.replace('\n','')
                element = element.strip('\n')
                sequence_len = len(element)
                ind = int((window_size-1)/2)
                element = 'X'*ind + element + 'X'*ind
                #print(element)
                for x in range(ind+1, len(element)-ind+1):
                        temp = element[x-ind-1:x+ind]
                        #print(len(temp))
                        #print(temp)
                        pattern_list.append(temp)
                        #print(pattern_list)
        #print(len(pattern_list))
        return pattern_list


def encode_input(patterns):

        central_location = int((window_size-1)/2)
        #print(central_location)
        X, y = [], []

        class_count = { 0:0, 1:0 }
        for pattern in patterns:
                # A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, Y, W
                tmp = np.zeros((window_size,21),dtype='int')
                #print(tmp)
                interacting = 0

                for i,residue in enumerate(pattern):
                        if i == central_location and residue.islower():
                                interacting = 1
                        if residue ==  'A'  or residue == 'a':
                                tmp[i][0] = 1
                        elif residue ==  'C'  or residue == 'c':
                                tmp[i][1] = 1
                        elif residue ==  'D'  or residue == 'd':
                                tmp[i][2] = 1
                        elif residue ==  'E'  or residue == 'e':
                                tmp[i][3] = 1
                        elif residue ==  'F'  or residue == 'f':
                                tmp[i][4] = 1
                        elif residue ==  'G'  or residue == 'g':
                                tmp[i][5] = 1
                        elif residue ==  'H'  or residue == 'h':
                                tmp[i][6] = 1
                        elif residue ==  'I'  or residue == 'i':
                                tmp[i][7] = 1
                        elif residue ==  'K'  or residue == 'k':
                                tmp[i][8] = 1
                        elif residue ==  'L'  or residue == 'l':
                                tmp[i][9] = 1
                        elif residue ==  'M'  or residue == 'm':
                                tmp[i][10] = 1
                        elif residue ==  'N'  or residue == 'n':
                                tmp[i][11] = 1
                        elif residue ==  'P'  or residue == 'p':
                                tmp[i][12] = 1
                        elif residue ==  'Q'  or residue == 'q':
                                tmp[i][13] = 1
                        elif residue ==  'R'  or residue == 'r':
                                tmp[i][14] = 1
                        elif residue ==  'S'  or residue == 's':
                                tmp[i][15] = 1
                        elif residue ==  'T'  or residue == 't':
                                tmp[i][16] = 1
                        elif residue ==  'V'  or residue == 'v':
                                tmp[i][17] = 1
                        elif residue ==  'Y'  or residue == 'Y':
                                tmp[i][18] = 1
                        elif residue ==  'W'  or residue == 'w':
                                tmp[i][19] = 1
                        elif residue ==  'X'  or residue == 'x':
                                tmp[i][20] = 1
                #print(tmp)
                #print(interacting)
                tmp = tmp.flatten()
                #print(tmp)
                X.append(tmp)
                y.append(interacting)
                class_count[interacting]+=1

        #print(X[0])
        #print(patterns[0])
        #print(len(X))
        #print("Total residue samples: ", len(y))
        #print("Class distribution: ", class_count)
        #print("As this is highly unbalanced, we are balancing the data by equalizing the number of positive and negative samples")


        # Resampling by choosing random indices from the negative dataset to make the number of positive and negative samples equal
        positive_dataset = []
        negative_dataset = []
        for i, val  in enumerate(y):
                if val == 0:
                        negative_dataset.append((X[i],val))
                else:
                        positive_dataset.append((X[i],val))
        r_indices = random.sample(range(0, len(negative_dataset)), len(positive_dataset))
        neg_samples = [negative_dataset[i] for i in r_indices]
        #print('New class distribution: ')
        #print("0:", len(neg_samples))
        #print(neg_samples[0])
        #print("1:", len(positive_dataset))
        #print(positive_dataset[0])
        X_new = []
        y_new = []
        for sample, label in neg_samples:
                X_new.append(sample)
                y_new.append(label)
        for sample, label in positive_dataset:
                X_new.append(sample)
                y_new.append(label)
        X_new = np.array(X_new)
        y_new = np.array(y_new,dtype='int')
        return X_new,y_new

parser = MyParser('Performs ATP binding interaction residue prediction using a Neural Netork Model for Window Size 17')
parser.add_argument('--i', type=str, help = 'input file, ex: abc.txt', required=True)
args = parser.parse_args()
filename = args.i


data = load_dataset(file_name)
X,y = encode_input(data)
y = np_utils.to_categorical(y, 2)
y = np.array(y,dtype='int')
print(X.shape)
model = Sequential()
model.add(Dense(16,activation='tanh',input_shape=(504,)))
#model.add(Dropout(0.2))
#model.add(Dense(154,activation='relu'))
#model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=Adam())
print(model.summary())
model.fit(X,y,batch_size=None,epochs=100, verbose=1, callbacks=None, validation_split=0.2)
