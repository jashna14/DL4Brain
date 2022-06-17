import numpy as np
from scipy.stats import pearsonr
from scipy import spatial


def pairwise_accuracy(actual, predicted, distance):
	""" To compute the Pairwise accuracy

	Args:
		actual: Numpy array of shape (num_samples x D) - Actual Targets
		predicted: Numpy array of shape (num_samples x D) - Predicted Targets
		distance: String - to calculate distance between actual and predicted sample - can be one of the following 'pearson'/'cosine'

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

            if distance == 'pearson':
	            if pearsonr(s1,b1)[0] + pearsonr(s2,b2)[0] > pearsonr(s1,b2)[0] + pearsonr(s2,b1)[0]:
	                true += 1

	        elif distance == 'cosine':
	        	if spatial.distance.cosine(s1,b1) + spatial.distance.cosine(s2,b2) < spatial.distance.cosine(s1,b2) + spatial.distance.cosine(s2,b1):
	                true += 1

    return(true/total)
    

def rank_accuracy(actual, predicted, distance):
	""" To compute the Rank accuracy

	Args:
		actual: Numpy array of shape (num_samples x D) - Actual Targets
		predicted: Numpy array of shape (num_samples x D) - Predicted Targets
		distance: String - to calculate distance between actual and predicted sample - can be one of the following 'pearson'/'cosine'

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

## pearson corr, RSA etc