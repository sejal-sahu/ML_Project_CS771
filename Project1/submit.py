import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def binaryTointerger(bits):
	x = 0
	for (i, j) in enumerate(bits):
		if(j):
	  		x += 2**(3 - i)
	return x

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	
	Z_train = Z_train.astype(bool)

  	# Seperating configBits, selectBits and response bits
	configBits, selectBits, response = np.hsplit(Z_train, [64, 72])
	response = response.ravel()

 	# Change config bits from 1/0 to 1 , -1 for better training;
	configBits = 1 - 2*configBits

	# Logistic Regression: 120 linear models for 120 pairs
	model = [[LogisticRegression(penalty = 'l2', C = 7, tol = 0.02, random_state = 0) for j in range (i)] for i in range (16)]

	# converting selectBits into integers to later use it to map models to respective data 
	m, n = selectBits.shape
	s1, s2 = np.zeros(shape = (m, ), dtype = int), np.zeros(shape = (m, ) , dtype = int);
	for i in range(m):
		s1[i] = binaryTointerger(selectBits[i, :4]);
		s2[i] = binaryTointerger(selectBits[i, 4:8]);


	# seperating dataset for 120 linear models.
	my_list = [[[] for j in range (i)] for i in range(16)]

	for i in range(m):
	# assume first xorro represents higher integer than the second xorros, wherever unture flip the respose bit
		if s1[i] < s2[i] :
			response[i] = not response[i]
		xorro1 = max(s1[i], s2[i])
		xorro2 = min(s1[i], s2[i])
		my_list[xorro1][xorro2].append(i)

	# training the 120 models seperately.
	for i in range(16):
		for j in range(i):
			if len(my_list[i][j]) != 0:
				targetConfig = configBits[my_list[i][j]]
				targetResponse = response[my_list[i][j]]
				model[i][j].fit(targetConfig,targetResponse)

	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
	
	X_tst = X_tst.astype(bool)

	#seperating configBits, selectBits and response bits
	configBits, selectBits, response = np.hsplit(X_tst, [64, 72])
	response = response.ravel()

	# change config bits from 1/0 to 1 , -1;
	configBits = 1 - 2*configBits

  	# converting selectBits into integers to later use it to map models to respective data
	m, n = selectBits.shape
	pred = np.empty(shape = (m, ), dtype = int)
	s1, s2 = np.zeros(shape = (m, ), dtype = int),np.zeros(shape = (m, ) , dtype = int);

	# store flip instances seperately
	fli = []

	for i in range(m):
		s1[i] = binaryTointerger(selectBits[i, :4]);
		s2[i] = binaryTointerger(selectBits[i, 4:8]);

	# seperating dataset for 120 linear models.
	my_list = [[[] for j in range (i)] for i in range(16)]

	for i in range(m):
		if s1[i] < s2[i] :
			fli.append(i)
		xorro1 = max(s1[i], s2[i])
		xorro2 = min(s1[i], s2[i])
		my_list[xorro1][xorro2].append(i)

	for i in range(16):
		for j in range(i):
			if len(my_list[i][j]) != 0:
				targetData = my_list[i][j]
				targetConfig = configBits[my_list[i][j]]
				pred[targetData] = model[i][j].predict(targetConfig)

	# flip the flip instances

	pred[fli] = 1 - pred[fli] 

	# Use this method to make predictions on test challenges

	return pred

if __name__ == "__main__":

    Z_train = np.genfromtxt("train.dat");
    model = my_fit(Z_train)

    Z_test = np.genfromtxt("test.dat");

    pred = my_predict(Z_test, model)

    acc = accuracy_score(Z_test[:, -1], pred)
    print(acc)
