import sys
'''Implementation of 'Identification of ATP binding residues of a protein from its primary sequence',
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-434.

Aditya Adhikary(2015007)
Mayank Kumar Pal(2015147)

We have implemented the SVM based classification of ATP binding residues from the available dataset at:

http://webs.iiitd.edu.in/raghava/atpint/atpdataset


Dependencies: numpy, sklearn, pandas

'''

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, f1_score
import random
import pandas as pd
import argparse

file_name = 'generated.txt'
window_size = 17 #default window size

class MyParser(argparse.ArgumentParser):
	def error(self,message):
		sys.stderr.write('Error: %s\n' %message)
		self.print_help()
		sys.exit()
'''
The following function loads the dataset from the ATP dataset given. It appends (window_size-1)/2
 X's or unknown residues at the end of each amino acid sequence, then breaks each sequence into window_size length
patterns and returns approximately 60,000 such patterns (same number as total number of residues).
'''
def load_dataset(file_name):

        pattern_list = []
        f1 = open(file_name,'r')
        line = f1.readlines()
        count = 0
        for element in line:
                count +=1
                element = element[1:]
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
        #print(count)
        return pattern_list

'''The following function converts the above patterns into (window_size*21) length vectors for the input to the SVM.
First, it labels each pattern as ATP interacting(1) or non-interacting(0) depending on whether the central residue is interacting or not.
Then, it goes through each pattern and replaces each residue by a 21-length binary vector with the position of 1
representing the residue. It then flattens the [window_size X 21] matrix to a single (window_size*21) length vector.
It also carries out balancing of data since there are too many negative samples.

'''
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
        X_new = sparse.csr_matrix(X_new) 
        return X_new,y_new

parser = MyParser(description="Runs the script  to perform ATP interacting residue preediction on amino sequences with window sizes 7 to 25")
parser.add_argument('--i', type = str, help = 'input file, ex. abc.txt', required = True)
parser.add_argument('--o', type = str, help = 'output file, ex. out.txt', required = True)
args = parser.parse_args()

filename = args.i
out_f = args.o
 

table = []
for win in range(7,25):
        window_size = win
        print("--------------------------------------------------------------\n")
        print("Performing ATP interacting residue prediction on amino acid sequences with window size = ", window_size)

        # Loading dataset and encoding to binary form
        data = load_dataset(file_name)

        X,y = encode_input(data)
        y = np.array(y,dtype='int')

        # Declaring model
        svm = SVC(kernel='poly', C=100.0,gamma = 0.1)


        # Stratified cross validation considers training set and test set in each iteration with approx equal percentage of class labels
        svm.fit(X,y)
        print("Performing Stratified 5-fold cross validation with svm, rbf kernel, gamma = 0.1, C=100.0 ...\n")

        #Cross_val_predict takes the prediction of a sample when it is considered as a part of the test set during the validation step
        y_preds = cross_val_predict(svm, X, y, cv=StratifiedKFold(n_splits=5,random_state=8), n_jobs=-1)

        #Performance measures
        acc = accuracy_score(y, y_preds)
        prec = precision_score(y, y_preds)
        rec = recall_score(y, y_preds)
        tn, fp, fn, tp = confusion_matrix(y, y_preds).ravel()
        tpr = tp / (tp + fn)
        fpr = fp/ (fp + tn)
        spec = tn / (tn+fp)
        f1 = f1_score(y, y_preds)
        mcc = matthews_corrcoef(y, y_preds)
        print('Accuracy: {}'.format(acc))
        print('Precision Score/ Positive Predictive Value = tp/(tp+fp) : {}'.format(prec))
        print('Recall Score/ Sensitivity/ True Positive Rate = tp/(tp+fn) :{}'.format(rec))
        print('Specificity/ True Negative Rate = tn/(tn+fp) : {}'.format(spec))
        print('F1 Score : {}'.format(f1))
        print('Mathews Correlation Coefficient : {}\n'.format(mcc))
        #print("--------------------------------------------------------------\n")
        row = np.array([win, acc, prec,rec,spec,f1,mcc,tpr,fpr])
        table.append(row)

        
# Printing and saving results to a csv file
print("\n---------------------------Performance Measures-----------------------------------------------\n")
df = pd.DataFrame(np.array(table), columns = ['Window Size', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC','TPR','FPR'])
print(df)
df.to_csv(out_f, index = False)




#data = load_dataset(file_name)
