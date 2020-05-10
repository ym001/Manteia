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
from .Preprocess import Preprocess

class Classification:
	r"""
		This is the class to classify text in categorie a NLP task.
		
		Args:
		
			model_name (:obj:`string`, optional, defaults to  'bert'):
				give the name of a model.
				
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
				
			labels (:obj:`float`, optional, defaults to None):
				A list of labels.
				 
		Example::
		
			from Manteia.Classification import Classification
			documents=['a text','text b']
			labels=['a','b']
			Classification(documents,labels)
			
		Attributes:
	"""
	def __init__(self,documents = [],labels = [],model=None,process=False,verbose=True):
		self.process   = process
		self.verbose   = verbose
		self.model     = model
		self.documents = documents
		self.labels    = labels
		if self.process:
			if self.verbose:
				print('Classification process.')
			pp               = Preprocess(documents=self.documents,labels=self.labels)
			self.list_labels = pp.list_labels
			self.documents   = pp.documents
			self.labels      = pp.labels
			self.load_model()
			dt_train ,dt_validation=self.process_text()
			self.model.configuration(dt_train)
			self.model.fit(dt_train,dt_validation)
			
	def test(self):
		return "Classification Mantéïa."
		
	def load_model(self):
		if self.model is not None:
			self.model = model
		else:
			self.model = Model(num_labels=len(self.list_labels))
		self.model.load_tokenizer()
		self.model.load_class()
		
	def process_text(self):
		train_text, validation_text, train_labels, validation_labels = train_test_split(self.documents,self.labels, random_state=2018, test_size=0.1)

		train_ids,train_masks           = encode_text(train_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
		validation_ids,validation_masks = encode_text(validation_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
		train_labels                    = encode_label(train_labels,self.list_labels)
		validation_labels               = encode_label(validation_labels,self.list_labels)

		dt_train          = Create_DataLoader_train(train_ids,train_masks,train_labels)
		dt_validation     = Create_DataLoader_train(validation_ids,validation_masks,validation_labels)
		return dt_train ,dt_validation 
		
	def predict(self,documents):
		r"""
		This is the description of the predict function of the Classification class.
		
		Args:
		
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
				 
		Example::
		
			from Manteia.Classification import Classification
			documents=['a text','text b']
			labels=['a','b']
			cl = Classification(documents,labels)
			print(cl.predict(documents[0]))

		"""
		inputs,masks   = encode_text(documents,self.model.tokenizer)
		predict_inputs = totensors(inputs)
		predict_masks  = totensors(masks)
		dt             = Create_DataLoader_predict(inputs=predict_inputs,masks=predict_masks)
		prediction     = self.model.predict(dt)
		prediction     = decode_label(prediction,self.list_labels)
		return prediction
		



