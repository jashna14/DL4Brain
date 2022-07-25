import numpy as np
from scipy.stats import pearsonr
import scipy.stats as stats
from scipy import spatial


def acc_2v2(actual, predicted):
	""" To compute the 2v2 accuracy

	Args:
		actual: Numpy array of shape (num_samples x D) - Actual Targets
		predicted: Numpy array of shape (num_samples x D) - Predicted Targets
	Return:
		float: The Pairwise accuracy

    """
	true = 0
	total = 0
	for i in range(0,len(actual)):
		for j in range(i+1, len(actual)):
			total += 1

			s1 = actual[i]
			s2 = actual[j]
			b1 = predicted[i]
			b2 = predicted[j]


			if spatial.distance.cosine(s1,b1) + spatial.distance.cosine(s2,b2) < spatial.distance.cosine(s1,b2) + spatial.distance.cosine(s2,b1):
				true += 1

	return(true/total)


def acc_pairwise(actual, predicted):
	""" To compute the 2v2 accuracy

	Args:
		actual: Numpy array of shape (num_samples x D) - Actual Targets
		predicted: Numpy array of shape (num_samples x D) - Predicted Targets
	Return:
		float: The Pairwise accuracy

    """
	true = 0
	total = 0
	for i in range(0,len(actual)):
		for j in range(i+1, len(actual)):
			total += 1

			s1 = actual[i]
			s2 = actual[j]
			b1 = predicted[i]
			b2 = predicted[j]


			if pearsonr(s1,b1)[0] + pearsonr(s2,b2)[0] > pearsonr(s1,b2)[0] + pearsonr(s2,b1)[0]:
				true += 1

	return(true/total)
    

def acc_rankBased(actual, predicted):
	""" To compute the Rank accuracy

	Args:
		actual: Numpy array of shape (num_samples x D) - Actual Targets
		predicted: Numpy array of shape (num_samples x D) - Predicted Targets

	Return:
		float: The Rank accuracy

	"""

	rank_sum = 0

	for i in range(0,len(predicted)):
		b1 = predicted[i]
		true_corr = 0
		corr_score = list()
		for j in range(0, len(actual)):
			s1 = actual[j]
			corr = pearsonr(s1,b1)[0]
			corr_score.append(corr)

			if(i == j):
				true_corr = corr
						
		corr_score.sort(reverse = True)
		rank = corr_score.index(true_corr) + 1
		rank_sum += rank
	    
	rank_sum = rank_sum/(len(actual))

	return(1 - ((rank_sum-1)/(len(actual)-1)))

def corr_pearson(actual, predicted):
    pearson = []
    for i in range(len(actual)):
        pearson_r, _  = pearsonr(actual[i],predicted[i])
        pearson.append(pearson_r)
    
    return np.array(person).mean()