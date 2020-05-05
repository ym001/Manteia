#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Model.py
#  
#  Copyright 2020 Yves <yves@mercadier>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split,KFold
import time
import datetime
import gc

#model'distilbert','albert','xlnet','roberta','camenbert','scibert'
class Model:
	def __init__(self,model_name ='bert',num_labels=None): # constructeur
		self.model_name = model_name
		self.batch_size = 32
		self.epochs = 4
		self.MAX_SEQ_LEN = 64
		self.num_labels=num_labels
	def test(self):
		return "Model Mantéïa."
	def load(self):
		# Load the tokenizer.
		print('Loading {} tokenizer...'.format(self.model_name))

		if self.model_name=='bert':
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

			# Load BertForSequenceClassification, the pretrained BERT model with a single 
			# linear classification layer on top. 
			self.model     = BertForSequenceClassification.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = self.num_labels, # The number of output labels--2 for binary classification.
			# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)
		if self.model_name=='distilbert':
			self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
			self.model     = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)

		if self.model_name=='albert':
			self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1', do_lower_case=True)
			self.model     = AlbertForSequenceClassification.from_pretrained("albert-base-v1",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
	
		if self.model_name=='xlnet':
			self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
			self.model     = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)

		if self.model_name=='roberta':
			self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
			self.model     = RobertaForSequenceClassification.from_pretrained("roberta-base",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)

		if self.model_name=='camenbert':
			self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
			self.model     = CamembertForSequenceClassification.from_pretrained("camembert-base",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='gpt2-medium':
			self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
			self.model     = GPT2LMHeadModel.from_pretrained('gpt2-medium')

		self.device()
		
	def device(self):
		# If there's a GPU available...
		if torch.cuda.is_available():    
			# Tell PyTorch to use the GPU.    
			self.device = torch.device("cuda")
			print('There are %d GPU(s) available.' % torch.cuda.device_count())
			print('We will use the GPU:', torch.cuda.get_device_name(0))

		else:
			print('No GPU available, using the CPU instead.')
			self.device = torch.device("cpu")

	def configuration(self,train_dataloader):
		self.model.cuda()
		self.optimizer = AdamW(self.model.parameters(),lr = 2e-5,eps = 1e-8)
		self.total_steps = len(train_dataloader) * self.epochs
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps = 0,num_training_steps = self.total_steps)

		
	def fit(self,train_dataloader,validation_dataloader):
		seed_val = 42
		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)

		loss_values = []

		for epoch_i in range(0, self.epochs):
    
			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
			print('Training...')

			t0 = time.time()
			total_loss = 0

			self.model.train()

			for step, batch in enumerate(train_dataloader):
				if step % 40 == 0 and not step == 0:
						elapsed = format_time(time.time() - t0)
						print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


				b_input_ids = batch[0].to(self.device)
				b_input_mask = batch[1].to(self.device)
				b_labels = batch[2].to(self.device)

				self.model.zero_grad()        

				if self.model_name != 'distilbert':
						outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
				else:
						outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
				loss = outputs[0]
				total_loss += loss.item()

				loss.backward()

				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

				self.optimizer.step()

				self.scheduler.step()

				avg_train_loss = total_loss / len(train_dataloader)            
    
				loss_values.append(avg_train_loss)

				print("")
				print("  Average training loss: {0:.2f}".format(avg_train_loss))
				print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        

				print("")
				print("Running Validation...")

				t0 = time.time()

				self.model.eval()

				eval_loss, eval_accuracy = 0, 0
				nb_eval_steps, nb_eval_examples = 0, 0

				for batch in validation_dataloader:
        
					batch = tuple(t.to(self.device) for t in batch)
        
					b_input_ids, b_input_mask, b_labels = batch
        
					with torch.no_grad():        

						if self.model_name != 'distilbert':
							outputs = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
						else:
							outputs = self.model(b_input_ids,attention_mask=b_input_mask)
						logits = outputs[0]

						logits = logits.detach().cpu().numpy()
						label_ids = b_labels.to('cpu').numpy()
        
					tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
					eval_accuracy += tmp_eval_accuracy

					nb_eval_steps += 1

				print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
				print("  Validation took: {:}".format(format_time(time.time() - t0)))

		print("")
		print("Training complete!")
		
	def predict(self,predict_dataloader):
		self.model.eval()
		prediction=[]
		for batch in predict_dataloader:
        
					batch = tuple(t.to(self.device) for t in batch)
        
					b_input_ids, b_input_mask = batch
        
					with torch.no_grad():        

						if self.model_name != 'distilbert':
							outputs = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
						else:
							outputs = self.model(b_input_ids,attention_mask=b_input_mask)
						tmp_logits = outputs[0]

						tmp_logits = tmp_logits.detach().cpu().numpy()
					prediction.extend(flat_prediction(tmp_logits))
		return prediction
	def fit_generation(self,text_loader):

		self.model.train()
		optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
		t_total = len(text_loader) // self.EPOCHS

		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.WARMUP_STEPS,num_training_steps=t_total)
		proc_seq_count = 0
		sum_loss = 0.0
		batch_count = 0
		tmp_jokes_tens = None

		for epoch in range(self.EPOCHS):
    
			print('EPOCH :'+str(epoch ))
    
			for idx,text in enumerate(text_loader):
        
				#################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
				joke_tens = torch.tensor(self.tokenizer.encode(text[0])).unsqueeze(0).to(self.device)
				#Skip sample from dataset if it is longer than MAX_SEQ_LEN
				if joke_tens.size()[1] > self.MAX_SEQ_LEN:
					continue
        
				#The first joke sequence in the sequence
				if not torch.is_tensor(tmp_jokes_tens):
					tmp_jokes_tens = joke_tens
					continue
				else:
					#The next joke does not fit in so we process the sequence and leave the last joke 
					#as the start for next sequence 
					if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > self.MAX_SEQ_LEN:
						work_jokes_tens = tmp_jokes_tens
						tmp_jokes_tens = joke_tens
					else:
						#Add the joke to sequence, continue and try to add more
						tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:,1:]], dim=1)
						continue
				################## Sequence ready, process it trough the model ##################
            
				outputs = model(work_jokes_tens, labels=work_jokes_tens)
				loss, logits = outputs[:2]                        
				loss.backward()
				sum_loss = sum_loss + loss.detach().data
                       
				proc_seq_count = proc_seq_count + 1
				if proc_seq_count == BATCH_SIZE:
					proc_seq_count = 0    
					batch_count += 1
					optimizer.step()
					scheduler.step() 
					optimizer.zero_grad()
					model.zero_grad()

				if batch_count == 100:
					print("sum loss :"+str(sum_loss))
					batch_count = 0
					sum_loss = 0.0


	def predict_generation(self,seed):
		self.model.to(self.device)
		self.model.eval()
		with torch.no_grad():
			joke_finished = False

			cur_ids   = torch.tensor(self.tokenizer.encode(seed)).unsqueeze(0).to(self.device)

			for i in range(200):
				outputs = self.model(cur_ids, labels=cur_ids)
				loss, logits = outputs[:2]
				softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
				if i < 3:
					n = 20
				else:
					n = 3
				next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
				cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(self.device) * next_token_id], dim = 1) # Add the last word to the running sequence

				if next_token_id in self.tokenizer.encode('<|endoftext|>'):
					joke_finished = True
					break
                                
			output_list = list(cur_ids.squeeze().to('cpu').numpy())
		return output_list

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)
		

def Create_DataLoader_train(inputs,masks,labels,batch_size=16):
		td = TensorDataset(totensors(inputs), totensors(masks), totensors(labels))
		rs = RandomSampler(td)
		return DataLoader(td, sampler=rs, batch_size=batch_size)

def Create_DataLoader_predict(predict_inputs,predict_masks,batch_size=16):
		predict_data = TensorDataset(totensors(predict_inputs), totensors(predict_masks))
		predict_sampler = RandomSampler(predict_data)
		return DataLoader(predict_data, sampler=predict_sampler, batch_size=batch_size)

def Create_DataLoader_generation(text):
		return DataLoader(TextDataset(text), batch_size=1, shuffle=True)

class TextDataset():
	def __init__(self,list_texts):
		super().__init__()

		self.joke_list = []
		end_of_text_token = "<|endoftext|>"
		for text in list_texts:
			self.joke_list.append(text+end_of_text_token)
        
	def __len__(self):
		return len(self.joke_list)

	def __getitem__(self, item):
		return self.joke_list[item]
		
def encode_text(sentences=None,tokenizer=None,MAX_SEQ_LEN=128):
		# Get the lists of sentences and their labels.

		print(' Original: ', sentences[0])

		# Print the sentence split into tokens.
		print('Tokenized: ', tokenizer.tokenize(sentences[0]))

		# Print the sentence mapped to token ids.
		print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

		# Tokenize all of the sentences and map the tokens to thier word IDs.
		input_ids = []

		for sent in sentences:
			encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
			input_ids.append(encoded_sent)

		#print('Max sentence length: ', max([len(sen) for sen in input_ids]))
		def pad_sequence(sequence=None,MAX_SEQ_LEN=None,pad='post'):
			pad_seq=[0]*MAX_SEQ_LEN
			if pad=='post':
				if len(sequence)<MAX_SEQ_LEN:
					pad_seq=pad_seq[len(sequence):]
					pad_seq=sequence+pad_seq
				else:
					pad_seq=pad_seq+sequence[:MAX_SEQ_LEN]
			if pad=='pre':
				if len(sequence)<MAX_SEQ_LEN:
					pad_seq=pad_seq[:MAX_SEQ_LEN-len(sequence)]
					pad_seq=pad_seq+sequence
				else:
					pad_seq=pad_seq+sequence[:MAX_SEQ_LEN]
			return pad_seq
		pad='post'
		#if self.model_name=='xlnet':
		#	pad='pre'
		#input_ids=[pad_sequence(sequence,self.MAX_SEQ_LEN,pad) for sequence in input_ids]

		#input_ids=[tokenizer.encode(text=sent,add_special_tokens=True,max_length=MAX_SEQ_LEN,pad_to_max_length=True) for sent in sentences]
		dico_input_and_mask = [tokenizer.encode_plus(text=sent,add_special_tokens=True,max_length=MAX_SEQ_LEN,pad_to_max_length=True,return_attention_mask=True) for sent in sentences]
		attention_masks = []
		input_ids       = []
		for dico in dico_input_and_mask:
			input_ids.append(dico['input_ids'])
			attention_masks.append(dico['attention_mask'])
		print('\nPadding/truncating all sentences to %d values...' % MAX_SEQ_LEN)

		#attention_masks = []

		# For each sentence...
		#for sent in input_ids:  
		#	att_mask = [int(token_id > 0) for token_id in sent]
		#	attention_masks.append(att_mask)

		return input_ids,attention_masks
		
def encode_label(labels,list_labels):
	def label_int(label):
			if label in list_labels:
				idx_label=list_labels.index(label)
			return idx_label
	labels_int=[]
	for lab in labels:
		labels_int.append(label_int(lab))
	return labels_int
	
def decode_label(labels_int,list_labels):
		labels_string = []
		for label in labels_int:
			str_label = list_labels[label]
			labels_string.append(str_label)
		return labels_string
		
def decode_text(output_list,tokenizer):
	return tokenizer.decode(output_list)

def totensors(inputs):
		return torch.tensor(inputs)
		
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_prediction(preds):
    return np.argmax(preds, axis=1).flatten()
