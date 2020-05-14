"""
.. module:: Preprocess
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from Manteia.Utils import load_animation

TEXT_COLUMN = 'texts'
LABEL_COLUMN = 'labels'
ID_COLUMN = 'id'

LANG=['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek','hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'turkish']
 
class Preprocess:
	r"""
		This is the class to preprocess text before task NLP.
   
		Args:
		
			lang='english',preprocess=True):

			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
			labels (:obj:`list`, optional, defaults to None):
				A list of labels.
			percentage (:obj:`float`, optional, defaults to 1.0):
				Percentage of the reduction data.
			size_by_nb_sample (:obj:`bool`, optional, defaults to False):
				Type of réduction by sample or by percentage.
			nb_sample (:obj:`int`, optional, defaults to None):
				Number of sample after reduction.
			path (:obj:`string`, optional, defaults to './Document/'):
				Path to save data object.
			lang (:obj:`string`, optional, defaults to 'english'):
				lang of stop word.
			preprocess (:obj:`bool`, optional, defaults to 1):
				make preprocess in init.
            
		Example::
		
			from Manteia.Preprocess import *
			import pandas as pd
			# Initializing a list of texts,labels
			documents=['a text','text b']
			# Initializing preprocess configuration
			pp=Preprocess(documents)
			pp.load()
			pp.df_documents=clean(pp.df_documents)
			print(pp.df_documents.head())
			
		Attributes:
	"""
	def __init__(self,documents=[],labels=[],percentage=1.0,nb_sample=0,path='./Document/',lang='english',preprocess=True,verbose=True):

		self.documents=documents
		self.labels=labels
		self.percentage=percentage
		self.nb_sample=nb_sample
		self.path=path
		self.lang=lang
		self.verbose=verbose
		
		if preprocess and documents!=[] and labels!=[]:
			#print('Preprocess.')
			load_animation('preprocess.',2)
			################
			for i in range(len(documents)):
				documents[i]=str(documents[i])
			for i in range(len(labels)):
				labels[i]=str(labels[i])
			################
			self.load()
			self.reduction()
			self.df_documents=clean(self.df_documents)
			self.list_labels=list_labels(self.df_labels[LABEL_COLUMN].values.tolist())
		
			self.documents=self.df_documents[TEXT_COLUMN].values.tolist()
			self.labels=self.df_labels[LABEL_COLUMN].values.tolist()
			
	def test(self):
		return "Preprocess Mantéïa."
		
	def load(self): # load data -> dataframe df
		if self.documents!=[]:
			self.df_documents=pd.DataFrame({TEXT_COLUMN:self.documents})
		if self.labels!=[]:
			self.df_labels  =pd.DataFrame({LABEL_COLUMN:self.labels})
			#multiclass
			#self.df_labels[LABEL_COLUMN] = self.df_labels[LABEL_COLUMN].apply(lambda x: x[0])

	def reduction(self,stratify=False):
		if self.nb_sample!=0:
			if self.nb_sample<self.df_documents.shape[0]:
				self.percentage=1-self.nb_sample*1.0/self.df_documents.shape[0]
		if self.percentage<1.0:
			if stratify:
				self.df_documents, self.df_test_documents, self.df_labels, self.df_test_labels = train_test_split(self.df_documents, self.df_labels, test_size=self.percentage, random_state=42,stratify=self.df_labels)
			else:
				self.df_documents, self.df_test_documents, self.df_labels, self.df_test_labels = train_test_split(self.df_documents, self.df_labels, test_size=self.percentage, random_state=42)
		
	def get_labels_int(self):
		return self.df_labels.values.tolist()
		
	def construct_id(self):
		return [id_idx for id_idx in range(len(self.df_labels))]

	def get_documents(self):
		return self.df_documents
		
	def get_labels(self):
		return self.df_labels

	def get_df(self):
		return pd.DataFrame({TEXT_COLUMN:self.df_documents[TEXT_COLUMN] , LABEL_COLUMN:self.df_labels[LABEL_COLUMN]})

	
def list_labels(labels):
		return list(np.sort(np.unique(np.array(labels)), axis=0))
'''
	def list_labels(self,labels):
		label=[]
		for l in labels:
				if l not in label:
					label.append(l)
		label.sort(reverse=False)
		return label
'''
		
def clean_stop_word(df,lang='english'):
		stop_unicode = stopwords.words(lang)
		#dictionary to string conversion
		stop=[str(w) for w in stop_unicode]
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))#stop word
		return df

def clean_html(df):
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: re.sub(re.compile('<.*?>'), '', x))#supprime balise html
		return df
		
def clean_contraction(df,lang='english'):
		#remove contraction
		def _get_contractions(contraction_dict):
			contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
			return contraction_dict, contraction_re
		if lang=='english':
			contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

		contractions, contractions_re = _get_contractions(contraction_dict)
		def replace_contractions(text):
			def replace(match):
				return contractions[match.group(0)]
			return contractions_re.sub(replace, text)
			
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: replace_contractions(x))
		return df
		
def clean_special_char(df):
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: re.sub(re.compile('[^a-zA-z0-9\s]'), ' ', x))#del special char
		return df
		
def clean_lower(df):
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x.lower())#passe en minuscule
		return df

def clean_number(df):
		def clean_numbers(x):
			if bool(re.search(r'\d', x)):
				x = re.sub('[0-9]{5,}', '#####', x)
				x = re.sub('[0-9]{4}', '####', x)
				x = re.sub('[0-9]{3}', '###', x)
				x = re.sub('[0-9]{2}', '##', x)
			return x
			
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: clean_numbers(x))
		return df
		
def clean_spaces(df):
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: re.sub("[ ]{2,}", " ", x))#del spaces
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x.strip())#del space at begin and end
		return df
		
def lemmatizer(df):
		lemmatizer = WordNetLemmatizer()
		df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
		return df
		
def clean(df,lang='english'):
		df=clean_stop_word(df,lang)
		df=clean_html(df)
		df=clean_contraction(df,lang)
		df=clean_special_char(df)
		df=clean_lower(df)
		df=clean_number(df)
		df=clean_spaces(df)
		df=lemmatizer(df)
		return df
		


def save_json(documents,labels,path):
		dataset=[]
		for doc,lab in zip(documents,labels):
			dataset.append([doc,[lab]])
		f = open(path+".json", "w")
		f.write(json.dumps(dataset))
		f.close()
