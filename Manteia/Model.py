"""
.. module:: Model
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""

import warnings
warnings.filterwarnings("ignore")

import os
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
    CamembertTokenizer,
    FlaubertTokenizer
)
from transformers import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import XLMForSequenceClassification
from transformers import XLNetForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import AlbertForSequenceClassification
from transformers import CamembertForSequenceClassification
from transformers import FlaubertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from Manteia.Utils import progress

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
	r"""
		This is the class to construct model.
		
		Args:
		
			model_name (:obj:`string`, optional, defaults to  'bert'):
				give the name of a model.
			num_labels (:obj:`int`, optional, defaults to  '0'):
				give the number of categorie for classification.
				

				 
		Example::
		
			from Manteia.Preprocess import Preprocess
			from Manteia.Model import Model,encode_text,encode_label,Create_DataLoader_train
			from sklearn.model_selection import train_test_split

			documents=['a text','text b']
			labels=['a','b']
			pp               = Preprocess(documents=documents,labels=labels)
			model       = Model(model_name=model_name,num_labels=len(pp.list_labels))
			model.load()

			train_text, validation_text, train_labels, validation_labels = train_test_split(pp.documents, pp.labels, random_state=2018, test_size=0.1)

			train_ids,train_masks           = encode_text(train_text,model.tokenizer,MAX_SEQ_LEN)
			validation_ids,validation_masks = encode_text(validation_text,model.tokenizer,MAX_SEQ_LEN)
			train_labels                    = encode_label(train_labels,pp.list_labels)
			validation_labels               = encode_label(validation_labels,pp.list_labels)

			dt_train          = Create_DataLoader_train(train_ids,train_masks,train_labels)
			dt_validation     = Create_DataLoader_train(validation_ids,validation_masks,validation_labels)
		
			model.configuration(dt_train)
			model.fit(dt_train,dt_validation)
			
		Attributes:
	"""
	def __init__(self,model_name ='bert',num_labels=0,MAX_SEQ_LEN = 128,early_stopping=False,path='./model',verbose=True): 
		
		self.model_name     = model_name
		self.early_stopping = early_stopping
		self.num_labels     = num_labels
		self.MAX_SEQ_LEN    = MAX_SEQ_LEN
		self.path           = path
		self.verbose        = verbose
		
		
			
		seed_val = 42
		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)
		
	def test(self):
		return "Model Mantéïa."
		
	def load_tokenizer(self):
		# Load the tokenizer.
		if self.verbose==True:
			print('Loading {} tokenizer...'.format(self.model_name))
		if self.model_name=='bert':
			#model_type='bert-base-uncased'
			model_type='bert-base-multilingual-cased'
			self.tokenizer = BertTokenizer.from_pretrained      (model_type, do_lower_case=True)
		if self.model_name=='distilbert':
			self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
		if self.model_name=='albert':
			self.tokenizer = AlbertTokenizer.from_pretrained    ('albert-base-v1', do_lower_case=True)	
		if self.model_name=='xlnet':
			self.tokenizer = XLNetTokenizer.from_pretrained     ('xlnet-base-cased', do_lower_case=True)
		if self.model_name=='roberta':
			self.tokenizer = RobertaTokenizer.from_pretrained   ('roberta-base', do_lower_case=True)
		if self.model_name=='camenbert':
			self.tokenizer = CamembertTokenizer.from_pretrained ('camembert-base', do_lower_case=True)
		if self.model_name=='flaubert':
			self.tokenizer = FlaubertTokenizer.from_pretrained  ('flaubert-base-uncased', do_lower_case=True)
		if self.model_name=='gpt2-medium':
			self.tokenizer = GPT2Tokenizer.from_pretrained      ('gpt2-medium')
			
	def load_class(self):
		# Load the tokenizer.
		if self.verbose==True:
			print('Loading {} class...'.format(self.model_name))
		num_labels = self.num_labels # The number of output labels
		if self.model_name=='bert':
			#model_type='bert-base-uncased'
			model_type='bert-base-multilingual-cased'
			# Load BertForSequenceClassification, the pretrained BERT model with a single 
			# linear classification layer on top. 
			self.model     = BertForSequenceClassification.from_pretrained(model_type, # Use the 12-layer BERT model, with an uncased vocab.
			# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)
		if self.model_name=='distilbert':
			self.model     = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='albert':
			self.model     = AlbertForSequenceClassification.from_pretrained    ("albert-base-v1",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='xlnet':
			self.model     = XLNetForSequenceClassification.from_pretrained     ("xlnet-base-cased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='roberta':
			self.model     = RobertaForSequenceClassification.from_pretrained   ("roberta-base",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='camenbert':
			self.model     = CamembertForSequenceClassification.from_pretrained ("camembert-base",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='flaubert':
			self.model     = FlaubertForSequenceClassification.from_pretrained  ("flaubert-base-uncased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		if self.model_name=='gpt2-medium':
			self.model     = GPT2LMHeadModel.from_pretrained('gpt2-medium')
		
	def devices(self):
		# If there's a GPU available...
		if torch.cuda.is_available():    
			# Tell PyTorch to use the GPU.    
			self.device = torch.device("cuda")
			if self.verbose==True:
				print('There are %d GPU(s) available.' % torch.cuda.device_count())
				print('We will use the GPU:', torch.cuda.get_device_name(0))

		else:
			if self.verbose==True:
				print('No GPU available, using the CPU instead.')
			self.device = torch.device("cpu")

	def configuration(self,train_dataloader,batch_size = 16,epochs = 4,n_gpu=1):
		self.batch_size  = batch_size
		self.epochs      = epochs
		self.n_gpu       = n_gpu
		#self.model.cuda()
		self.optimizer   = AdamW(self.model.parameters(),lr = 2e-5,eps = 1e-8)
		self.total_steps = len(train_dataloader) * self.epochs
		self.scheduler   = get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps = 0,num_training_steps = self.total_steps)
		self.devices()

		
	def fit(self,train_dataloader,validation_dataloader):
		
		self.model.to(self.device)
		#self.model.cuda()
		
		if self.early_stopping:
			self.es=EarlyStopping(path=self.path)
			
		if self.n_gpu > 1:
			self.model = torch.nn.DataParallel(self.model)
            
		loss_values = []

		for epoch_i in range(0, self.epochs):
			if self.verbose==True:

				print("")
				print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
				print('Training :')

			t0 = time.time()
			total_loss = 0

			self.model.train()

			for step, batch in enumerate(train_dataloader):
				'''
				if step % 40 == 0 and not step == 0:
						elapsed = format_time(time.time() - t0)
						if self.verbose==True:
							print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
				'''

				b_input_ids = batch[0].to(self.device)
				b_input_mask = batch[1].to(self.device)
				b_labels = batch[2].to(self.device)

				self.model.zero_grad()        

				if self.model_name != 'distilbert':
						outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
				else:
						outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

				
				loss = outputs[0]
				if self.n_gpu > 1:
					loss = loss.mean()
				loss.backward()
				total_loss += loss.item()


				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				self.optimizer.step()
				self.scheduler.step()

				avg_train_loss = total_loss / len(train_dataloader)            
    
				loss_values.append(avg_train_loss)
				progress(count=step+1, total=len(train_dataloader))

			if self.verbose==True:
				print("")
				print("  Average training loss: {0:.2f}".format(avg_train_loss))
				print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
				print("Validation :")

			t0 = time.time()

			self.model.eval()

			tab_logits = None
			tab_labels = None

			
			for step,batch in enumerate(validation_dataloader):
        
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

						
						if tab_logits is None:tab_logits=np.argmax(logits, axis=1)
						else:tab_logits=np.append(tab_logits,np.argmax(logits, axis=1), axis=0)
						if tab_labels is None:tab_labels=label_ids
						else:tab_labels=np.append(tab_labels,label_ids, axis=0)
					progress(count=step+1, total=len(validation_dataloader))

						
			acc_validation=accuracy(tab_logits, tab_labels)
			if self.verbose==True:
				print("")
				print("  Accuracy: {0:.2f}".format(acc_validation))
				print("  Validation took: {:}".format(format_time(time.time() - t0)))

			if self.early_stopping:
				self.es(acc_validation, self.model,self.device)
                
				if self.es.early_stop:
					if self.verbose==True:
						print("Early stopping")
					break
		if self.verbose==True:
			print("")
			print("Training complete!")
	"""
	p_type='class' or 'probability' or 'logits'
	"""
	def predict(self,predict_dataloader,p_type='class'):
		if self.early_stopping:
			#by torch
			#pour charger uniquement la classe du modèle!
			self.load_class()
			self.model.load_state_dict(torch.load(self.path+'state_dict_validation.pt'))

			#by transformer
			#self.model.from_pretrained(self.path)
			if self.verbose==True:
				print('loading model early...')
				
		#self.model.cuda()
		self.model.to(self.device)

		self.model.eval()
		predictions = None

		for batch in predict_dataloader:
        
					batch = tuple(t.to(self.device) for t in batch)
        
					b_input_ids, b_input_mask = batch
        
					with torch.no_grad():        

						if self.model_name != 'distilbert':
							outputs = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
						else:
							outputs = self.model(b_input_ids,attention_mask=b_input_mask)
						logits = outputs[0]

						logits = logits.detach().cpu()
					if p_type=='class':
						if predictions is None:predictions=np.argmax(logits.numpy(), axis=1)
						else:predictions=np.append(predictions,np.argmax(logits.numpy(), axis=1), axis=0)
					if p_type=='logits':
						if predictions is None:predictions=logits.numpy()
						else:predictions=np.append(predictions,logits.numpy(), axis=0)
					if p_type=='probability':
						if predictions is None:predictions=torch.softmax(logits, dim=1).numpy()
						else:predictions=np.append(predictions,torch.softmax(logits, dim=1).numpy(), axis=0)
		return predictions
		
	def fit_generation(self,text_loader):
		self.model.to(self.device)

		self.model.train()
		optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
		t_total = len(text_loader) // self.EPOCHS

		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.WARMUP_STEPS,num_training_steps=t_total)
		proc_seq_count = 0
		sum_loss = 0.0
		batch_count = 0
		tmp_jokes_tens = None

		for epoch in range(self.EPOCHS):
			if self.verbose==True:
				print('EPOCH :'+str(epoch ))
  
			for step,text in enumerate(text_loader):
        
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
            
				outputs = self.model(work_jokes_tens, labels=work_jokes_tens)
				loss, logits = outputs[:2]                        
				loss.backward()
				sum_loss = sum_loss + loss.detach().data
                       
				proc_seq_count = proc_seq_count + 1
				if proc_seq_count == self.batch_size:
					proc_seq_count = 0    
					batch_count += 1
					optimizer.step()
					scheduler.step() 
					optimizer.zero_grad()
					self.model.zero_grad()

				if batch_count == 100:
					if self.verbose==True:
						print("sum loss :"+str(sum_loss))
					batch_count = 0
					sum_loss = 0.0
				progress(count=step+1, total=len(text_loader))


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
		
	def save(self,file_name):
		if not os.path.isdir(self.path):
			# define the name of the directory to be created
			try:
				os.mkdir(self.path)
			except OSError:
				print ("Creation of the directory %s failed" % self.path)
			else:
				print ("Successfully created the directory %s " % self.path)
		self.model.to(torch.device('cpu'))
		torch.save(self.model.module.state_dict(),self.path+file_name+'.pt')
		model.to(self.device)
		
	def load(self,file_name):
			self.load_class()
			self.model.load_state_dict(torch.load(self.path+file_name))
			self.model.to(self.device)

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
		
#a supprimer!!!!!!
def Create_DataLoader_predict(inputs,masks,batch_size=16):
		td = TensorDataset(totensors(inputs), totensors(masks))
		ss = SequentialSampler(td)
		return DataLoader(td, sampler=ss, batch_size=batch_size)

def Create_DataLoader_generation(text,batch_size=16):
		return DataLoader(TextDataset(text), batch_size=batch_size, shuffle=True)

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
		
def encode_text(sentences=None,tokenizer=None,MAX_SEQ_LEN=128,verbose=False):
		# Get the lists of sentences and their labels.
		if verbose==True:
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
		if verbose==True:
			print('\nPadding/truncating all sentences to %d values...' % MAX_SEQ_LEN)

		return input_ids,attention_masks
		
def encode_label(labels,list_labels):
	
	def label_int(label):
			if label in list_labels:
				idx_label=list_labels.index(label)
			return idx_label
	labels_int=[]
	for lab in labels:
		labels_int.append(label_int(lab))
	return np.array(labels_int)
	
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
def accuracy(preds, labels):
	return (preds == labels).mean()

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=2, delta=0,path=None, verbose=True):
		"""
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 2
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.acc_validation_min = 0
		self.delta = delta
		self.path=path
		if os.path.isfile(self.path+'state_dict_validation.pt'):
			os. remove(self.path+'state_dict_validation.pt') 

	def __call__(self, acc_validation , model,device_model):
        
		score = acc_validation

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(acc_validation, model,device_model)
		elif score < self.best_score:
			self.counter += 1
			if self.verbose==True:
				print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			if self.verbose==True:
				print('Save model : {} out of {}'.format(self.counter,self.patience))
			self.best_score = score
			self.save_checkpoint(acc_validation, model,device_model)
			self.counter = 0

	def save_checkpoint(self, acc_validation, model,device_model):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.acc_validation_min,acc_validation))
		if not os.path.isdir(self.path):
			# define the name of the directory to be created
			try:
				os.mkdir(self.path)
			except OSError:
				print ("Creation of the directory %s failed" % self.path)
			else:
				print ("Successfully created the directory %s " % self.path)
		#save by torch
		device = torch.device('cpu')
		model.to(device)
		print(type(model))
		torch.save(model.module.state_dict(),self.path+'state_dict_validation.pt')
		
		model.to(device_model)
		#save by transformer
		#model.save_pretrained(self.path)
		self.acc_validation_min = acc_validation

'''
bert-base-uncased,bert-large-uncased,bert-base-cased,bert-large-cased,bert-base-multilingual-uncased,bert-base-multilingual-cased,bert-base-chinese
,bert-base-german-cased,bert-large-uncased-whole-word-masking,bert-large-cased-whole-word-masking,bert-large-uncased-whole-word-masking-finetuned-squad
,bert-large-cased-whole-word-masking-finetuned-squad,bert-base-cased-finetuned-mrpc,bert-base-german-dbmdz-cased
,bert-base-german-dbmdz-uncased,bert-base-japanese,bert-base-japanese-whole-word-masking,bert-base-japanese-char,bert-base-japanese-char-whole-word-masking
,bert-base-finnish-cased-v1,bert-base-finnish-uncased-v1,bert-base-dutch-cased


openai-gpt,GPT-2,gpt2,gpt2-medium,gpt2-large,gpt2-xl,Transformer-XL,transfo-xl-wt103

xlnet-base-cased,xlnet-large-cased,
xlm-mlm-en-2048,xlm-mlm-ende-1024,xlm-mlm-enfr-1024,xlm-mlm-enro-1024,xlm-mlm-xnli15-1024,xlm-mlm-tlm-xnli15-1024,xlm-clm-enfr-1024,xlm-clm-ende-1024
,xlm-mlm-17-1280,xlm-mlm-100-1280

roberta-base,roberta-large,roberta-large-mnli,distilroberta-base,roberta-base-openai-detector,roberta-large-openai-detector

,distilbert-base-uncased,distilbert-base-uncased-distilled-squad,distilbert-base-cased,distilbert-base-cased-distilled-squad

distilgpt2,distilbert-base-german-cased,distilbert-base-multilingual-cased

ctrl

camembert-base,

albert-base-v1,albert-large-v1,albert-xlarge-v1,albert-xxlarge-v1,albert-base-v2,albert-large-v2,albert-xlarge-v2,albert-xxlarge-v2

t5-small,t5-base,t5-large,t5-3B,t5-11B

,xlm-roberta-base,xlm-roberta-large

flaubert-small-cased,flaubert-base-uncased,flaubert-base-cased,flaubert-large-cased

bart-large,bart-large-mnli,bart-large-cnn,mbart-large-en-ro

DialoGPT-small,DialoGPT-medium,DialoGPT-large

reformer-crime-and-punishment
'''
