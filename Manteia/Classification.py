"""
.. module:: useful_1
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Andrew Carter <andrew@invalid.com>


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
			# Initializing a list of texts,labels
			documents=['a text','text b']
			labels=['a','b']
			Classification(documents,labels)
			
		Attributes:
	"""
	def __init__(self,documents = [],labels = [],model=None,process=False,verbose=True):
		
		if documents!=[] and labels!=[]:
			pp               = Preprocess(documents=documents,labels=labels)
			self.list_labels = pp.list_labels
			
		if model!=None:
			self.model = model
		else:
			self.model = Model(num_labels=len(pp.list_labels))

		if process:
			print('Process...')
			self.model.load()

			train_text, validation_text, train_labels, validation_labels = train_test_split(pp.documents, pp.labels, random_state=2018, test_size=0.1)

			train_ids,train_masks           = encode_text(train_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
			validation_ids,validation_masks = encode_text(validation_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
			train_labels                    = encode_label(train_labels,pp.list_labels)
			validation_labels               = encode_label(validation_labels,pp.list_labels)

			dt_train          = Create_DataLoader_train(train_ids,train_masks,train_labels)
			dt_validation     = Create_DataLoader_train(validation_ids,validation_masks,validation_labels)
		
			self.model.configuration(dt_train)
			self.model.fit(dt_train,dt_validation)
			
	def test(self):
		return "Classification Mantéïa."
		
	def predict(self,documents):
		inputs,masks   = encode_text(documents,self.model.tokenizer)
		predict_inputs = totensors(inputs)
		predict_masks  = totensors(masks)
		dt             = Create_DataLoader_predict(inputs=predict_inputs,masks=predict_masks)
		prediction     = self.model.predict(dt)
		prediction     = decode_label(prediction,self.list_labels)
		return prediction
		



