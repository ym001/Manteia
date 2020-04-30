#https://github.com/aniruddhachoudhury/BERT-Tutorials/blob/master/Blog%202/BERT_Fine_Tuning_Sentence_Classification.ipynb
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split,KFold
import time
import datetime
import gc
############
from .Model import Model
from .Preprocess import Preprocess

class Classification:
	def __init__(self,model_name ='bert',documents = None,labels = None,data = None): # constructeur
		self.epochs=1
		self.batch_size=16
		self.MAX_SEQ_LEN=64
		self.model_name=model_name
		
		if documents!=None and labels!=None:
			pp=Preprocess(documents,labels)
			self.num_labels=len(pp.list_labels)
			pp.labels_to_int()
		
			self.labels=pp.get_labels_int()
			#model'distilbert','albert','xlnet','roberta','camenbert','scibert'
			######################
			self.model=Model(num_labels=self.num_labels)
			self.model.load()
			self.tokenizer=self.model.tokenizer#!
			self.device=self.model.device#!
		
			input_ids,attention_masks=self.model.encode_text(pp.documents,self.model.tokenizer)

			self.train_inputs, self.validation_inputs, self.train_labels, self.validation_labels = train_test_split(input_ids, self.labels, random_state=2018, test_size=0.1)
			self.train_masks, self.validation_masks, _, _ = train_test_split(attention_masks, self.labels,random_state=2018, test_size=0.1)
			#print(self.train_inputs)
			self.train_inputs=self.model.totensors(self.train_inputs)
			self.train_masks=self.model.totensors(self.train_masks)
			self.train_labels=self.model.totensors(self.train_labels)
			self.validation_inputs=self.model.totensors(self.validation_inputs)
			self.validation_masks=self.model.totensors(self.validation_masks)
			self.validation_labels=self.model.totensors(self.validation_labels)

			self.Create_DataLoader()
			#self.train_dataloader =test.Create_DataLoader(self.train_inputs,self.train_masks,self.labels)

			td,vd=self.train_dataloader,self.validation_dataloader
			#self.train_dataloader =test.Create_DataLoader(self.train_inputs,self.train_masks,self.labels)
			#self.validation_dataloader=test.Create_DataLoader(self.validation_inputs,self.validation_masks,self.labels)
		
			self.model.configuration(td)
			self.model.fit(td,vd)
	def test(self):
		return "Classification Mantéïa."
	def Create_DataLoader(self):
		train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
		train_sampler = RandomSampler(train_data)
		self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

		validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
		validation_sampler = SequentialSampler(validation_data)
		self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)


