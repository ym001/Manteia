
import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split,KFold
import time
import datetime
import gc
############
from .Model import *
from .Preprocess import Preprocess

class Generation:
	r"""
		This is the class to gnerate text in categorie a NLP task.
		
		Args:
			model_name (:obj:`string`, optional, defaults to  'bert'):
				give the name of a model.
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
			labels (:obj:`float`, optional, defaults to None):
				A list of labels.
				 
		Example::
			from Manteia.Classification import Classification
			# Initializing a list of texts,labels
			documents=['a text','text b']
			labels=['a','b']
			Classification(documents,labels)
		Attributes:
	"""
	def __init__(self,model_name ='gpt2-medium',documents = None,labels = None):


		model               = Model(model_name =model_name)
		model.load()
		text_loader         = Create_DataLoader_generation(documents)
		model.BATCH_SIZE    = 16
		model.EPOCHS        = 10
		model.LEARNING_RATE = 3e-5
		model.WARMUP_STEPS  = 500
		model.MAX_SEQ_LEN   = 400
		
		model.fit_generation(text_loader)
		
		output              = model.predict_generation('joke')
		output_text         = decode_text(output,model.tokenizer)
		print(output_text)

		
	def test(self):
		return "Generation Mantéïa."
		

		

