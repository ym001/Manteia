"""
.. module:: Augmentation
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
import numpy as np
import random
from nltk.corpus import wordnet
import collections
import math

#import nltk
#nltk.download('wordnet')

class Augmentation:
	
	r"""
		This is the class to do data augmentation.
		
		Args:
		
				
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
				
			labels (:obj:`float`, optional, defaults to None):
				A list of labels.

			dataset_name (:obj:`string`, optional, defaults to ''):
				Name of the dataset.

			path (:obj:`string`, optional, defaults to ''):
				Path to save the report.
				 
		Example::
		
			from Manteia.Statistic import Statistic
			documents=['a text','text b']
			labels=['a','b']
			Statistic(documents,labels)
			
		Attributes:
	"""
	def __init__(self,documents=[],labels=[],strategy='daia',verbose=True):
		
		self.documents    = documents
		self.labels       = labels
		self.verbose      = verbose
		if verbose:
				print('Augmentation %s.' % strategy)
		if strategy=='eda':
			self.documents_augmented,self.labels_augmented = eda(self.documents,self.labels)
		if strategy=='uda':
			self.documents_augmented,self.labels_augmented = eda(self.documents,self.labels)
		if strategy=='pyramid':
			self.documents_augmented,self.labels_augmented = pyramid(self.documents,self.labels)
		
	def test(self):
		return "Mantéïa Augmentation."

def uda(documents,labels):
	documents_augmented=[]
	labels_augmented=[]
	
	data_stats=get_data_stats(documents)
	token_prob=0.9
	op = TfIdfWordRep(token_prob, data_stats)

	for text,label in zip(documents,labels):
		text_aug=op(text)
		documents_augmented.append(text_aug)
		labels_augmented.append(label)
	return documents_augmented,labels_augmented

#https://github.com/google-research/uda/blob/master/text/augmentation/word_level_augment.py
def pyramid(documents,labels,level):
	r"""
		This function compute DAIA.
		
		Args:
			documents
			labels
			level
		return
			documents_augmented
			labels_augmented

		Example::
			
	"""
	documents_augmented=[]
	labels_augmented=[]
	if level < 2:level=2
	if level > 5:level=5
	for text,label in zip(documents,labels):
		text_list,label_list=split_text(text,label,level)
		documents_augmented  = documents_augmented+text_list
		labels_augmented = labels_augmented+label_list
	return documents_augmented,labels_augmented
	
def get_data_stats(texts):
  """Compute the IDF score for each word. Then compute the TF-IDF score."""
  word_doc_freq = collections.defaultdict(int)
  # Compute IDF
  for text in texts:
    cur_word_dict = {}
    cur_sent = text.split(' ')
    for word in cur_sent:
      cur_word_dict[word] = 1
    for word in cur_word_dict:
      word_doc_freq[word] += 1
  idf = {}
  for word in word_doc_freq:
    idf[word] = math.log(len(texts) * 1. / word_doc_freq[word])
  # Compute TF-IDF
  tf_idf = {}
  for text in texts:
    cur_word_dict = {}
    cur_sent = text.split(' ')
    for word in cur_sent:
      if word not in tf_idf:
        tf_idf[word] = 0
      tf_idf[word] += 1. / len(cur_sent) * idf[word]
  return {
      "idf": idf,
      "tf_idf": tf_idf,
  }
  
class EfficientRandomGen(object):
  """A base class that generate multiple random numbers at the same time."""

  def reset_random_prob(self):
    """Generate many random numbers at the same time and cache them."""
    cache_len = 100000
    self.random_prob_cache = np.random.random(size=(cache_len,))
    self.random_prob_ptr = cache_len - 1

  def get_random_prob(self):
    """Get a random number."""
    value = self.random_prob_cache[self.random_prob_ptr]
    self.random_prob_ptr -= 1
    if self.random_prob_ptr == -1:
      self.reset_random_prob()
    return value

  def get_random_token(self):
    """Get a random token."""
    token = self.token_list[self.token_ptr]
    self.token_ptr -= 1
    if self.token_ptr == -1:
      self.reset_token_list()
    return token
    
class TfIdfWordRep(EfficientRandomGen):
  """TF-IDF Based Word Replacement."""

  def __init__(self, token_prob, data_stats):
    super(TfIdfWordRep, self).__init__()
    self.token_prob = token_prob
    self.data_stats = data_stats
    self.idf = data_stats["idf"]
    self.tf_idf = data_stats["tf_idf"]
    tf_idf_items = data_stats["tf_idf"].items()
    tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
    self.tf_idf_keys = []
    self.tf_idf_values = []
    for key, value in tf_idf_items:
      self.tf_idf_keys += [key]
      self.tf_idf_values += [value]
    self.normalized_tf_idf = np.array(self.tf_idf_values)
    self.normalized_tf_idf = (self.normalized_tf_idf.max()
                              - self.normalized_tf_idf)
    self.normalized_tf_idf = (self.normalized_tf_idf
                              / self.normalized_tf_idf.sum())
    self.reset_token_list()
    self.reset_random_prob()

  def get_replace_prob(self, all_words):
    """Compute the probability of replacing tokens in a sentence."""
    cur_tf_idf = collections.defaultdict(int)
    for word in all_words:
      cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
    replace_prob = []
    for word in all_words:
      replace_prob += [cur_tf_idf[word]]
    replace_prob = np.array(replace_prob)
    replace_prob = np.max(replace_prob) - replace_prob
    replace_prob = (replace_prob / replace_prob.sum() *
                    self.token_prob * len(all_words))
    return replace_prob

  def __call__(self, example):

    all_words = example.split(' ')

    replace_prob = self.get_replace_prob(all_words)
    all_words = self.replace_tokens(
        all_words,
        replace_prob[:len(all_words)]
        )

    return " ".join(all_words)

  def replace_tokens(self, word_list, replace_prob):
    """Replace tokens in a sentence."""
    for i in range(len(word_list)):
      if self.get_random_prob() < replace_prob[i]:
        word_list[i] = self.get_random_token()
    return word_list

  def reset_token_list(self):
    cache_len = len(self.tf_idf_keys)
    token_list_idx = np.random.choice(
        cache_len, (cache_len,), p=self.normalized_tf_idf)
    self.token_list = []
    for idx in token_list_idx:
      self.token_list += [self.tf_idf_keys[idx]]
    self.token_ptr = len(self.token_list) - 1
    #print("sampled token list: {:s}".format(" ".join(self.token_list)))

def eda(documents,labels):
	documents_augmented=[]
	labels_augmented=[]

	for document,label in zip(documents,labels):
		text_list,label_list = eda_text(document,label)
		documents_augmented  = documents_augmented+text_list
		labels_augmented     = labels_augmented+label_list
	return documents_augmented,labels_augmented
	
def eda_text(text,label):
	text_list,label_list=[],[]
	
	#pour decoupage en word
	word_list_1=text.split(' ')
	#inversion de deux mot
	idx_1 = random.randint(0,len(word_list_1)-1) 
	idx_2 = random.randint(0,len(word_list_1)-1) 
	word_list_1[idx_1],word_list_1[idx_2] = word_list_1[idx_2],word_list_1[idx_1]
	text_list = [' '.join(word_list_1)]
	label_list= [label]
	#suppression d'un mot mot
	word_list_2=text.split(' ')
	idx_3 = random.randint(0,len(word_list_2)-1) 
	del word_list_2[idx_1]
	text_list.append(' '.join(word_list_2))
	label_list.append(label)
	#Synonym Replacement
	word_list_3=text.split(' ')
	idx_4 = random.randint(0,len(word_list_3)-1) 
	if len(wordnet.synsets(word_list_3[idx_4]))>0:
			idx_synonym=random.randint(0,len(wordnet.synsets(word_list_3[idx_4]))-1)
			synonym = wordnet.synsets(word_list_3[idx_4])[idx_synonym].lemma_names()[0]
			if synonym!=word_list_3[idx_4]:
				word_list_3[idx_4]=synonym
				text_list.append(' '.join(word_list_2))
				label_list.append(label)
	#Random Insertion (RI)
	word_list_4=text.split(' ')
	idx_5 = random.randint(0,len(word_list_4)-1) 
	idx_6 = random.randint(0,len(word_list_4)-1) 
	if len(wordnet.synsets(word_list_4[idx_5]))>0:
			idx_synonym=random.randint(0,len(wordnet.synsets(word_list_4[idx_5]))-1)
			synonym = wordnet.synsets(word_list_4[idx_5])[idx_synonym].lemma_names()[0]
			if synonym!=word_list_4[idx_5]:
				word_list_4.insert(idx_6, synonym)
				text_list.append(' '.join(word_list_2))
				label_list.append(label)
	return text_list,label_list

def split_text(text,label,level=3):
	text_list,label_list=[],[]
	
	decoup_1a = int(0.05*len(text))
	decoup_1b = int(0.95*len(text))
	decoup_2 = int(len(text)/2)
	decoup_3 = int(len(text)/3)
	decoup_4 = int(len(text)/4)
	decoup_5 = int(len(text)/5)

	if level >=1 :
		text_list  = text_list+[text[decoup_1a:decoup_1b]]
		label_list = label_list+[label]
	if level >=2 :
		text_list  = text_list+[text[:decoup_2],text[decoup_2:]]
		label_list = label_list+[label,label]
	if level >=3 :
		text_list  = text_list+[text[:decoup_3],text[decoup_3:2*decoup_3],text[2*decoup_3:]]
		label_list = label_list+[label,label,label]
	if level >=4 :
		text_list  = text_list+[text[:decoup_4],text[decoup_4:2*decoup_4],text[2*decoup_4:3*decoup_4],text[3*decoup_4:]]
		label_list = label_list+[label,label,label,label]
	if level >=5 :
		text_list  = text_list+[text[:decoup_5],text[decoup_5:2*decoup_5],text[2*decoup_5:3*decoup_5],text[3*decoup_5:4*decoup_5],text[4*decoup_5:]]
		label_list = label_list+[label,label,label,label,label]

	return text_list,label_list

	

