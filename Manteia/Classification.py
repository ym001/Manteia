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

				 
		Example 1::

		
			from Manteia.Classification import Classification 
			from Manteia.Model import Model 
			
			documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.'
			,'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
			labels = ['funny','not funny']
			
			model = Model(model_name ='roberta')
			cl=Classification(model,documents,labels,process_classif=True)
			
			>>>Training complete!
	"""
	def __init__(self,model=None,documents_train = [],labels_train = [],documents_test = [],labels_test = [],process_classif=False,verbose=True):
		
		self.process_classif = process_classif
		self.verbose         = verbose
		self.model           = model
		self.documents_train = documents_train
		self.labels_train    = labels_train
		self.documents_test  = documents_test
		self.labels_test     = labels_test
		
		if self.process_classif and self.documents_train!=[] and self.labels_train!=[]:
			
			self.list_labels     = list_labels(self.labels_train)
			self.process()
			
			
	def test(self):
		
		return "Classification Mantéïa."


	def process(self):
		"""
		Example 2::
		
			from Manteia.Classification import Classification 
			from Manteia.Preprocess import list_labels 
			from Manteia.Model import Model 
			
			documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.'
			,'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
			labels = ['funny','not funny']
			
			model = Model(model_name ='roberta')
			cl=Classification(model,documents,labels)
			cl.list_labels     = list_labels(labels)
			cl.process()
			print(cl.predict(documents[:2]))
			>>>['funny', 'funny']
		"""
		self.load_model()
		dt_train ,dt_validation=self.process_text()
		self.model.configuration(dt_train)
		self.model.fit(dt_train,dt_validation)
		if self.documents_test != []:
			predictions_test=self.predict(self.documents_test)
			if self.labels_test !=[]:
				if self.verbose:
					print("accuracy : ".format(accuracy(predictions_test, self.labels_test)))
					
	def load_model(self):
		"""
		Example 3::
		
			from Manteia.Classification import Classification 
			from Manteia.Preprocess import list_labels 
			
			documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.'
			,'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
			labels = ['funny','not funny']
			
			cl=Classification(documents_train = documents,labels_train = labels)
			cl.list_labels     = list_labels(labels)
			cl.load_model()
			cl.model.devices()
			print(cl.predict(documents[:2]))
			>>>['funny', 'funny']
		"""
		if self.model is None:
			self.model = Model()
		self.model.load_type()
		self.model.load_tokenizer()
		self.model.num_labels=len(self.list_labels)
		self.model.load_class()
		
	

	def process_text(self):
		r"""
		This is the description of the process_text function.
		
		Example 4::
		
			from Manteia.Classification import Classification 
			from Manteia.Preprocess import list_labels 
			
			documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.'
			,'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
			labels = ['funny','not funny']
			
			cl=Classification(documents_train = documents,labels_train = labels)
			cl.list_labels     = list_labels(labels)
			cl.load_model()
			dt_train ,dt_validation=cl.process_text()
			cl.model.configuration(dt_train)
			cl.model.fit(dt_train,dt_validation)

			>>>Training complete!
		"""
		train_text, validation_text, train_labels, validation_labels = train_test_split(self.documents_train,self.labels_train, random_state=2018, test_size=0.1)

		train_ids,train_masks           = encode_text(train_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
		validation_ids,validation_masks = encode_text(validation_text,self.model.tokenizer,self.model.MAX_SEQ_LEN)
		train_labels                    = encode_label(train_labels,self.list_labels)
		validation_labels               = encode_label(validation_labels,self.list_labels)

		dt_train          = Create_DataLoader(train_ids,train_masks,train_labels)
		dt_validation     = Create_DataLoader(validation_ids,validation_masks,validation_labels)
		return dt_train ,dt_validation 
		
	def predict(self,documents):
		r"""
		This is the description of the predict function.
		
		Args:
		
			documents (:obj:`list`, optional, defaults to None):
				A list of documents (str).
				 
					 
		Example 5::
		
			from Manteia.Classification import Classification 
			from Manteia.Model import Model 
			
			documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.'
			,'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
			labels = ['funny','not funny']
			
			model = Model(model_name ='roberta')
			cl=Classification(model,documents,labels,process_classif=True)
			print(cl.predict(documents[:2]))
			
			>>>['funny', 'funny']
		"""
		inputs,masks   = encode_text(documents,self.model.tokenizer)
		predict_inputs = totensors(inputs)
		predict_masks  = totensors(masks)
		dt             = Create_DataLoader(inputs=predict_inputs,masks=predict_masks)
		prediction     = self.model.predict(dt)
		prediction     = decode_label(prediction,self.list_labels)
		return prediction
		



