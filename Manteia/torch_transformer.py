#https://github.com/aniruddhachoudhury/BERT-Tutorials/blob/master/Blog%202/BERT_Fine_Tuning_Sentence_Classification.ipynb

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

import sys

######################
import torch

import numpy as np # linear algebra
import math
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split,KFold
import gc
############
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (
    WEIGHTS_NAME,PreTrainedModel,
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
############
#batch_size = 16
batch_size = 8

MAX_SEQ_LEN = 128
#MAX_SEQ_LEN = 64
PATH_INIT='/home/mercadier/model/init/'
PATH_EARLY='/home/mercadier/model/early_stopping/'
path="/home/mercadier/these/resultat/result_expe.txt"
fichier = open(path, "w")
fichier.write('rapport\n')
fichier.close()
# Number of training epochs (authors recommend between 2 and 4)
epochs = 10
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

early_stopping=True
############

sentences = ['er ert gg','fss vv hh']
labels = ['z','v']
num_labels=2
##############
model_name ='bert'
#model_name = 'distilbert'
#model_name = 'albert'
#model_name='xlnet'
#model_name = 'roberta'
#model_name = 'camenbert'
#model_name = 'scibert'
large=True
# Load the tokenizer.
print('Loading {} tokenizer...'.format(model_name))

def load_model(model_name=None):
	if model_name=='bert':
		name='bert-base-uncased'
		if large==True:
			name='bert-large-uncased'

		tokenizer = BertTokenizer.from_pretrained(name, do_lower_case=True)

		# Load BertForSequenceClassification, the pretrained BERT model with a single 
		# linear classification layer on top. 
		model = BertForSequenceClassification.from_pretrained(name, # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = num_labels, # The number of output labels--2 for binary classification.
			# You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
		)
		class_model=BertForSequenceClassification
	if model_name=='distilbert':
		tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
		model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		class_model=DistilBertForSequenceClassification
		
	if model_name=='albert':
		name='albert-base-v2'
		if large==True:
			nama='albert-xxlarge-v2'
		tokenizer = AlbertTokenizer.from_pretrained(name, do_lower_case=True)
		model = AlbertForSequenceClassification.from_pretrained(name,num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		class_model=AlbertForSequenceClassification
		
	if model_name=='xlnet':
		name='xlnet-base-cased'
		if large==True:
			name='xlnet-large-cased'
		tokenizer = XLNetTokenizer.from_pretrained(name, do_lower_case=True)
		model = XLNetForSequenceClassification.from_pretrained(name,num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		class_model=XLNetForSequenceClassification
		
	if model_name=='roberta':
		name='roberta-base'
		if large==True:
			name='roberta-large'
		tokenizer = RobertaTokenizer.from_pretrained(name, do_lower_case=True)
		model = RobertaForSequenceClassification.from_pretrained(name,num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		class_model=RobertaForSequenceClassification
		
	if model_name=='camenbert':
		tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
		model = CamembertForSequenceClassification.from_pretrained("camembert-base",num_labels = num_labels,output_attentions = False,output_hidden_states = False,)
		class_model=CamembertForSequenceClassification
		
	if model_name=='scibert':
		model_pretrained='/home/mercadier/model/scibert/scibert_scivocab_uncased/'
		tokenizer = BertTokenizer.from_pretrained(model_pretrained, do_lower_case=True)
		#config a tester ou a enlever
		#config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
		model = BertForSequenceClassification.from_pretrained(model_pretrained,num_labels = num_labels,output_attentions = False,output_hidden_states = False,config=config )
		class_model=BertForSequenceClassification
	return model,tokenizer,class_model
	


# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    

        
def predict(model,dataloader):

			model.eval()

			# Tracking variables 
			accuracy = 0
			nb_steps = 0

			# Evaluate data for one epoch
			for batch in dataloader:
        
				# Add batch to GPU
				batch = tuple(t.to(device) for t in batch)
        
				# Unpack the inputs from our dataloader
				b_input_ids, b_input_mask, b_labels = batch
        
				# Telling the model not to compute or store gradients, saving memory and
				# speeding up validation
				with torch.no_grad():        

					# Forward pass, calculate logit predictions.
					# This will return the logits rather than the loss because we have
					# not provided labels.
					# token_type_ids is the same as the "segment ids", which 
					# differentiates sentence 1 and 2 in 2-sentence tasks.
					# The documentation for this `model` function is here: 
					# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
					if model_name != 'distilbert':
						outputs = model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)
					else:
						outputs = model(b_input_ids,attention_mask=b_input_mask)
				# Get the "logits" output by the model. The "logits" are the output
				# values prior to applying an activation function like the softmax.
				logits = outputs[0]

				# Move logits and labels to CPU
				logits = logits.detach().cpu().numpy()
				label_ids = b_labels.to('cpu').numpy()
        
				# Calculate the accuracy for this batch of test sentences.
				tmp_accuracy = flat_accuracy(logits, label_ids)
				# Accumulate the total accuracy.
				accuracy += tmp_accuracy

				# Track the number of batches
				nb_steps += 1
			# Report the final accuracy for this validation run.
			accuracy=accuracy/nb_steps
			return accuracy
			
# Get the lists of sentences and their labels.
'''
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

print('Max sentence length: ', max([len(sen) for sen in input_ids]))
def pad_sequence(sequence=None,MAX_SEQ_LEN=None,pad='post'):
	#pad_seq=np.zeros((MAX_SEQ_LEN,), dtype=int)
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
if model_name=='xlnet':
	pad='pre'
input_ids=[pad_sequence(sequence,MAX_SEQ_LEN,pad) for sequence in input_ids]
'''


# Use train_test_split to split our data into train and validation sets for
# training
from sklearn.model_selection import train_test_split

# Use 90% for training and 10% for validation.
#train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
# Do the same for the masks.
#train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,random_state=2018, test_size=0.1)

# Convert all inputs and labels into torch tensors, the required datatype 
# for our model.


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here.


# Tell pytorch to run this model on the GPU.
#model.cuda()
'''
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
'''

from transformers import get_linear_schedule_with_warmup

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
#for model_name in ['bert','distilbert','albert','xlnet','roberta','scibert']:
for model_name in ['bert','albert','xlnet','roberta']:
	model,tokenizer,class_model=load_model(model_name)
	model.save_pretrained(PATH_INIT)
	model.save_pretrained(PATH_EARLY)

	input_ids=[tokenizer.encode(text=sent,add_special_tokens=True,max_length=MAX_SEQ_LEN,pad_to_max_length=True) for sent in sentences]
	print(input_ids[0])
	#print('\nPadding/truncating all sentences to %d values...' % MAX_SEQ_LEN)
	#print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
	# Create attention masks
	attention_masks = []
	# For each sentence...
	for sent in input_ids:    
		# Create the attention mask.
		#   - If a token ID is 0, then it's padding, set the mask to 0.
		#   - If a token ID is > 0, then it's a real token, set the mask to 1.
		att_mask = [int(token_id > 0) for token_id in sent]
    
		# Store the attention mask for this sentence.
		attention_masks.append(att_mask)
	print(attention_masks[0])
	input_ids = torch.tensor(input_ids)
	labels = torch.tensor(labels)
	attention_masks = torch.tensor(attention_masks)


	
	#validation croisée
	xfold = np.zeros(len(train))
	skf = KFold(n_splits=p.k_fold)
	skf.get_n_splits(xfold)
	acc_cross_test=[]
	cpt_passe=0
	rapport=''
	for idx_train, idx_test in skf.split(xfold):
		print('Passe : {}'.format(cpt_passe))
		cpt_passe+=1
		#inversion des indices de validation croisée.
		idx_train, idx_test=idx_test,idx_train
		idx_validation=random.sample(list(idx_train), int(len(idx_train)/10))
		idx_train=[idx for idx in idx_train if idx not in idx_validation]

		# Create the DataLoader for our training set.
		train_data = TensorDataset(input_ids[idx_train], attention_masks[idx_train], labels[idx_train])
		train_sampler = RandomSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

		# Create the DataLoader for our test set.
		test_data = TensorDataset(input_ids[idx_test], attention_masks[idx_test], labels[idx_test])
		test_sampler = SequentialSampler(test_data)
		test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

		# Create the DataLoader for our validation set.
		validation_data = TensorDataset(input_ids[idx_validation], attention_masks[idx_validation], labels[idx_validation])
		validation_sampler = SequentialSampler(validation_data)
		validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

		# Total number of training steps is number of batches * number of epochs.
		total_steps = len(train_dataloader) * epochs

		model = class_model.from_pretrained(PATH_INIT)
		model.cuda()

		#vient de simple transformer
		warmup_ratio=0.06
		#warmup_ratio=0.12
		warmup_steps = math.ceil(total_steps * warmup_ratio)

		# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
		# I believe the 'W' stands for 'Weight Decay fix"
		optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
 
		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps,#0,  Default value in run_glue.py
                                            num_training_steps = total_steps)
		if early_stopping:
			es = EarlyStopping(patience=2, verbose=True,path=PATH_EARLY)
		model.train()
		# Store the average loss after each epoch so we can plot them.
		loss_values = []
		# For each epoch...
		for epoch_i in range(0, epochs):
    
			# ========================================
			#               Training
			# ========================================
    
			# Perform one full pass over the training set.

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			print('Training...')

			# Measure how long the training epoch takes.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_loss = 0

			# Put the model into training mode. Don't be mislead--the call to 
			# `train` just changes the *mode*, it doesn't *perform* the training.
			# `dropout` and `batchnorm` layers behave differently during training
			# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
			model.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):

				# Progress update every 40 batches.
				if step % 40 == 0 and not step == 0:
					# Calculate elapsed time in minutes.
					elapsed = format_time(time.time() - t0)
            
					# Report progress.
					print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
		
				# Unpack this training batch from our dataloader. 
				#
				# As we unpack the batch, we'll also copy each tensor to the GPU using the 
				# `to` method.
				#
				# `batch` contains three pytorch tensors:
				#   [0]: input ids 
				#   [1]: attention masks
				#   [2]: labels 
				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)

				# Always clear any previously calculated gradients before performing a
				# backward pass. PyTorch doesn't do this automatically because 
				# accumulating the gradients is "convenient while training RNNs". 
				# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
				model.zero_grad()        

				# Perform a forward pass (evaluate the model on this training batch).
				# This will return the loss (rather than the model output) because we
				# have provided the `labels`.
				# The documentation for this `model` function is here: 
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				if model_name != 'distilbert':
					outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
				else:
					outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
				# The call to `model` always returns a tuple, so we need to pull the 
				# loss value out of the tuple.
				loss = outputs[0]

				# Accumulate the training loss over all of the batches so that we can
				# calculate the average loss at the end. `loss` is a Tensor containing a
				# single value; the `.item()` function just returns the Python value 
				# from the tensor.
				total_loss += loss.item()
        
				# Perform a backward pass to calculate the gradients.
				loss.backward()

				# Clip the norm of the gradients to 1.0.
				# This is to help prevent the "exploding gradients" problem.
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

				# Update parameters and take a step using the computed gradient.
				# The optimizer dictates the "update rule"--how the parameters are
				# modified based on their gradients, the learning rate, etc.
				optimizer.step()

				# Update the learning rate.
				scheduler.step()

			# Calculate the average loss over the training data.
			avg_train_loss = total_loss / len(train_dataloader)            
			
			# Store the loss value for plotting the learning curve.
			loss_values.append(avg_train_loss)
		
			print("")
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
			print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.

			print("")
			print("Running Validation...")

			t0 = time.time()
			accuracy_validation=predict(model,validation_dataloader)
			# during evaluation.
			if early_stopping:
				es(accuracy_validation, model)
                 
				if es.early_stop:
					print("Early stopping")
					break
			else:
				print("  Accuracy  validation: {:}".format(accuracy_validation))
				print("  validation took: {:}".format(format_time(time.time() - t0)))
			print("")
			
		model = class_model.from_pretrained(PATH_EARLY)
		model.cuda()
		accuracy_test=predict(model,test_dataloader)
		acc_cross_test.append(accuracy_test)
		print("  Accuracy  test: {:}".format(accuracy_test))
		del model
		del optimizer
		del scheduler
		gc.collect()
		filelist = [ f for f in os.listdir(PATH_EARLY)]
		for f in filelist:
			os.remove(os.path.join(PATH_EARLY, f))
	rapport+="{}-{} Accuracy  test: {}-{:}\n".format(p.jeux,model_name,np.mean(acc_cross_test),acc_cross_test)
	fichier = open(path, "a")
	fichier.write(rapport)
	fichier.close()

print("Training complete!")

print("{} acc_cross_test{}:{}".format(model_name,acc_cross_test,np.mean(acc_cross_test)))
print("End of script.")


