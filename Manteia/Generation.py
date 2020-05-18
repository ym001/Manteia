"""
.. module:: Generation
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split,KFold
import time
import datetime
import gc
############
from Manteia.Model import *
from Manteia.Preprocess import Preprocess

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
		

			from Manteia.Generation import Generation 
			from Manteia.Dataset import Dataset
			from Manteia.Model import *
	
			ds=Dataset('Short_Jokes')

			model       = Model(model_name ='gpt2-medium')
			text_loader = Create_DataLoader_generation(ds.documents_train[:3000])
			model.load_tokenizer()
			model.load_class()
			model.devices()
			model.configuration(text_loader)
	
			gn=Generation(model)
	
			gn.model.fit_generation(text_loader)
			output      = model.predict_generation('What did you expect ?')
			output_text = decode_text(output,model.tokenizer)
			print(output_text)

	"""
	def __init__(self,model = None,documents = None,seed = None):

		if model is None:self.model = Model(model_name ='gpt2-medium')
		else : self.model=model
		#model.load()
		self.model.BATCH_SIZE    = 16
		self.model.EPOCHS        = 2
		self.model.LEARNING_RATE = 3e-5
		self.model.WARMUP_STEPS  = 500
		self.model.MAX_SEQ_LEN   = 400
		if documents!=None:
			text_loader         = Create_DataLoader_generation(documents)
			model.fit_generation(text_loader)
		if seed!=None:
			output              = model.predict_generation(seed)
			output_text         = decode_text(output,model.tokenizer)
			print(output_text)

		
	def test(self):
		return "Generation Mantéïa."
		
'''
https://huggingface.co/transformers/main_classes/model.html
see exemple!!!!
'''
		

