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
from Model import Model
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (
    WEIGHTS_NAME,
    BertTokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
    AlbertTokenizer,
    CamembertTokenizer
)
from transformers import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import XLMForSequenceClassification
from transformers import XLNetForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import AlbertForSequenceClassification
from transformers import CamembertForSequenceClassification

class Classification:
	def __init__(self,model_name ='bert',data=None): # constructeur
		self.epochs=1
		self.batch_size=16
		self.MAX_SEQ_LEN=64
		self.model_name=model_name
		self.num_labels=len(data.list_labels)
		data.labels_to_int()
		self.labels=data.get_labels_int()
		#model'distilbert','albert','xlnet','roberta','camenbert','scibert'
		######################
		test=Model(num_labels=self.num_labels)
		test.load()
		self.model=test.model
		self.tokenizer=test.tokenizer
		self.device=test.device
		
		input_ids,attention_masks=test.encode_text(data.documents,test.tokenizer)

		self.train_inputs, self.validation_inputs, self.train_labels, self.validation_labels = train_test_split(input_ids, self.labels, random_state=2018, test_size=0.1)
		self.train_masks, self.validation_masks, _, _ = train_test_split(attention_masks, self.labels,random_state=2018, test_size=0.1)
		
		self.train_inputs=test.totensors(self.train_inputs)
		self.train_masks=test.totensors(self.train_masks)
		self.train_labels=test.totensors(self.train_labels)
		self.validation_inputs=test.totensors(self.validation_inputs)
		self.validation_masks=test.totensors(self.validation_masks)
		self.validation_labels=test.totensors(self.validation_labels)

		self.Create_DataLoader()
		#self.train_dataloader =test.Create_DataLoader(self.train_inputs,self.train_masks,self.labels)

		td,vd=self.train_dataloader,self.validation_dataloader
		#self.train_dataloader =test.Create_DataLoader(self.train_inputs,self.train_masks,self.labels)
		#self.validation_dataloader=test.Create_DataLoader(self.validation_inputs,self.validation_masks,self.labels)
		
		test.configuration(td)
		test.fit(td,vd)
		
	def Create_DataLoader(self):
		train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
		train_sampler = RandomSampler(train_data)
		self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

		validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
		validation_sampler = SequentialSampler(validation_data)
		self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)


