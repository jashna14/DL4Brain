import numpy as np
from sklearn.linear_model import Ridge
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import json
import random



def RidgeReg_CrossVal(X, Y, k):
	""" To train a model using k-fold cross-validation ridge regression

	Args:
		X: Numpy array of shape (num_samples x D1) - Training sample Input
		Y: Numpy array of shape (num_samples x D2) - Training sample Expected Output 
		k: Int - NO. of folds

	Return:
		Two numpy arrays with actual and predicted outputs of shape (num_samples x D2)

	"""



	dataset_X = np.array(X.copy())
    dataset_Y = np.array(Y.copy())
    kf = KFold(n_splits=k)
    
    actual = []
    predicted = []

    for train_index, test_index in kf.split(dataset_X):

        X_train, X_test = dataset_X[train_index], dataset_X[test_index]
        y_train, y_test = dataset_Y[train_index], dataset_Y[test_index]
            
        model = Ridge(alpha=1.0)
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        actual.extend(y_test)
        predicted.extend(y_pred)


    return np.array(actual),np.array(predicted)


## verision of MLP?
