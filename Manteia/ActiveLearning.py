"""
.. module:: ActiveLearning
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
import numpy as np
import math
from operator import itemgetter

class RandomSampling():
	"""
	A random sampling query strategy baseline.
	"""

	def __init__(self,verbose=False):
		self.verbose=verbose
		if self.verbose:
			print('Randomsampling')
		
	def query(self, unlabeled_idx,nb_question):
		random.shuffle(unlabeled_idx) 
		selected_indices= unlabeled_idx[:nb_question]
		return selected_indices
		
class UncertaintyEntropySampling():
	"""
	The basic uncertainty sampling query strategy, querying the examples with the top entropy.
	"""

	def __init__(self,verbose=True):
		self.verbose=verbose
		if self.verbose:
			print('UncertaintyEntropySampling')

	def query(self,predictions,unlabeled_idx,nb_question):
		entropie=[]
		for prediction,idx in zip(predictions,unlabeled_idx):
			summ=0
			for proba in prediction:
				if proba>0:
					summ=summ-proba*math.log(proba)
			entropie.append((summ,idx))
		entropie_trie=sorted(entropie, key=itemgetter(0),reverse=True)
		idx_entropie=[tup[1] for tup in entropie_trie[:nb_question]]
		if self.verbose:
			print(entropie_trie[:nb_question])
		return idx_entropie




