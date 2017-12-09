# Improvement and Implementation of Paper 'Identification of ATP binding residues of a protein from its primary sequence'

Paper link : https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-10-434

Project Website Link : https://mynkpl1998.github.io/atppredict/

# Dependencies 
* Python 3
* Numpy
* sk-learn
* Pandas
* Keras
* Tensorflow or Theano

# How to Run ?
1. For SVM 
```
python3 classifier.py --i generated.txt --o out.txt
```

2. For Neural Network 
``` 
python3 nnet.py --i generated.txt
```

# Theory

We have implemented the SVM based classification of ATP binding residues from the available dataset at:http://webs.iiitd.edu.in/raghava/atpint/atpdataset.

For each window size in the range [7,25], first, the script loads the amino acid sequences from the ATP
dataset given.
It appends (window_size-1)/2 X's or unknown residues at the end of each amino acid sequence, then breaks
each sequence into window_size length patterns and returns approximately 60,000 such patterns (same number
as total number of residues).

Then, a function converts the above patterns into (window_size*21) length vectors for the input to the SVM. 
First, it labels each pattern as ATP interacting(1) or non-interacting(0) depending on whether the central
residue is interacting or not. 

Afterwards, it goes through each pattern and replaces each residue by a 21-length binary vector with the
position of 1 representing the residue. 
It then flattens the [window_size X 21] matrix to a single (window_size*21) length vector.
It also carries out balancing of data since there are too many negative samples.

It then declares the svm model and performs Stratified KFold Cross Validation, which considers training set and test set in each iteration with approx equal percentage of class labels. 
After getting back the predicted labels, it calculates different performance measures such as Accuracy,
Precision, Recall etc and prints the results.

# Improvements

Earilier the number of ATP binding protiens were very few (168 only). We derived the more ATP binding protiens using CD-Hit. Now we have 223 primary Sequences.
We trained it using SVM and Neural Network. 
And a improvement 6% is made from original implementation. Earlier the accuracy was about 64% now we have 70% accurate model.
Neural Network failed to give comparable accuracy as the training data was not adequate for deep Neural Nets.


