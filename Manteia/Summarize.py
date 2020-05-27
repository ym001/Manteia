#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. module:: Classification
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
from .Model import *
from .Preprocess import Preprocess,list_labels

class Summarize:
	r"""
		This is the class to summarize text.
		
		Args:
		
			model (:obj:`Model`, optional, defaults to  'bert'):
				give the name of a model.
				
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
				
			labels (:obj:`float`, optional, defaults to None):
				A list of labels.

				 
		Example 1::

		
			from Manteia.Summarize import Summarize
			
			
	"""
	def __init__(self,model=None,documents = [],verbose=True):
		
		self.process_classif = process_classif
		self.verbose         = verbose
		self.model           = model
		self.documents_train = documents_train
		self.labels_train    = labels_train
		self.documents_test  = documents_test
		self.labels_test     = labels_test

		self.load_model()
		inputs=self.process_text()
		print(self.predict(inputs))
					
	def load_model(self):
		"""
		Example 3::
		
			from Manteia.Summarize import Summarize 
			
		"""
		if self.model is None:
			self.model = Model(model_name ='bart',model_type='bart-large-cnn',task='summarize')
		self.model.load_type()
		self.model.load_tokenizer()
		self.model.load_class()
		
	

	def process_text(self):
		r"""
		This is the description of the process_text function.
		
		Example 4::
		
			from Manteia.Summarize import Summarize
		"""
		inputs = self.model.tokenizer.batch_encode_plus(self.documents, max_length=1024, return_tensors='pt')
		return inputs
		
	def predict(self,inputs):
		r"""
		This is the description of the predict function.
		
		Args:
		
			documents (:obj:`list`, optional, defaults to None):
				A list of documents (str).
				 
					 
		Example 5::
		
			from Manteia.Summarize import Summarize 
			
		"""
		summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
		summary     = [self.model.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
		return summary
		



