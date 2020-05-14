"""
.. module:: Dataset
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""

from .Preprocess import Preprocess
import numpy as np
from nltk.corpus import reuters,brown,webtext
from sklearn.datasets import fetch_20newsgroups
import urllib.request
import zipfile
import bz2
import wget
import csv

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from Manteia.Utils import bar_progress

class Dataset:
	
	r"""
		This is the class to give datasets.
		
		Args:

			dataset_name (:obj:`string`, optional, defaults to ''):
				Name of the dataset.
				 
		Example::

			

		Attributes:
	"""
	def __init__(self,name='20newsgroups',path='./dataset',verbose=True):

		self.name=name
		self.path=path
		self.verbose=verbose
		
		self.load()
		
	def test(self):
		return "Mantéïa Dataset."

	def load(self):
		if self.name=="20newsgroups":
			self.load_20newsgroups()
			
		if self.name=="SST-2":
			self.load_SST_2()
			
		if self.name=="SST-B":
			self.load_SST_B()

		if self.name=="drugscom":
			self.load_drugscom()
			
		if self.name=="pubmed_rct20k":
			self.load_pubmed_rct20k()
			
		if self.name=="yelp":
			self.load_yelp()
			
		if self.name=="trec":
			self.load_trec()
			
		if self.name=="agnews":
			self.load_agnews()
			
		if self.name=="DBPedia":
			self.load_DBPedia()
		if self.name=="Amazon Review Full":
			self.load_Amazon_Review_Full()
		if self.name=="Amazon Review Polarity":
			self.load_Amazon_Review_Polarity()
		if self.name=="Sogou News":
			self.load_Sogou_News()
		if self.name=="Yahoo! Answers":
			self.load_Yahoo_Answers()
		if self.name=="Yelp Review Full":
			self.load_Yelp_Review_Full()
		if self.name=="Yelp Review Polarity":
			self.load_Yelp_Review_Polarity()
			
	def load_20newsgroups(self):
		if self.verbose:
			print('Downloading 20newsgroups...')
		#categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space']
		categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		self.documents_train = twenty_train.data
		self.labels_train=[]
		for i in range(len(twenty_train.target)):
			self.labels_train.append(categorie[twenty_train.target[i]])
		
	def load_Yelp_Review_Polarity(self):
		
		self.path_dir = os.path.join(self.path,'yelp_review_polarity')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['yelp_review_polarity00.zip','yelp_review_polarity01.zip','yelp_review_polarity02.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
				
	def load_Yelp_Review_Full(self):
		
		self.path_dir = os.path.join(self.path,'yelp_review_full')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['yelp_review_full00.zip','yelp_review_full01.zip','yelp_review_full02.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row

	def load_Yahoo_Answers(self):
		"""
		Example Yahoo_Answers::

			from Manteia.Dataset import Dataset

			ds=Dataset('Yahoo! Answers')

			print('Test : ')
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])
		"""
		self.path_dir = os.path.join(self.path,'yahoo_answers')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['yahoo_answers00.zip','yahoo_answers01.zip','yahoo_answers02.zip','yahoo_answers03.zip',
						'yahoo_answers04.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
			
		self.path_classes = os.path.join(self.path_dir,'classes.txt')
		classes=['']
		if os.path.isfile(self.path_classes):
			fi = open(self.path_classes, "r")
			rows = fi.readlines()
			for row in rows:
				classes.append(row.strip())
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train=construct_sample(self.path_train,classes)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test=construct_sample(self.path_test,classes)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
	"""
	
	"""
	def load_Sogou_News(self):
		
		self.path_dir = os.path.join(self.path,'sogou_news')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['sogou_news00.zip','sogou_news01.zip','sogou_news02.zip','sogou_news03.zip',
						'sogou_news04.zip','sogou_news05.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
		self.path_classes = os.path.join(self.path_dir,'classes.txt')
		classes=['']
		if os.path.isfile(self.path_classes):
			fi = open(self.path_classes, "r")
			rows = fi.readlines()
			for row in rows:
				classes.append(row.strip())
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train = construct_sample(self.path_train,classes)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test   = construct_sample(self.path_test,classes)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row

	"""
	
	"""
	def load_Amazon_Review_Polarity(self):
		
		self.path_dir = os.path.join(self.path,'amazon_review_polarity')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['amazon_review_polarity00.zip','amazon_review_polarity01.zip','amazon_review_polarity02.zip','amazon_review_polarity03.zip',
						'amazon_review_polarity04.zip','amazon_review_polarity05.zip','amazon_review_polarity06.zip','amazon_review_polarity07.zip',
						'amazon_review_polarity08.zip','amazon_review_polarity09.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row

	def load_Amazon_Review_Full(self):
		
		self.path_dir = os.path.join(self.path,'amazon_review_full')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!

		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

			file_list=['amazon_review_full00.zip','amazon_review_full01.zip','amazon_review_full02.zip','amazon_review_full03.zip',
						'amazon_review_full04.zip','amazon_review_full05.zip','amazon_review_full06.zip','amazon_review_full07.zip',
						'amazon_review_full08.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
		
	def load_DBPedia(self):
		"""
		Example DBPedia::

			from Manteia.Dataset import Dataset

			ds=Dataset('DBPedia')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test = [],[]
		self.path_dir = os.path.join(self.path,'DBPedia')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!

		if not os.path.isdir(self.path_dir):
			
			if self.verbose:
				print('Downloading DBPedia.')
			url='https://github.com/ym001/Dune/raw/master/datasets/DBPedia.zip'
			download_and_extract(url, self.path)
			
		self.path_classes = os.path.join(self.path_dir,'classes.txt')
		classes=['']
		if os.path.isfile(self.path_classes):
			fi = open(self.path_classes, "r")
			rows = fi.readlines()
			for row in rows:
				classes.append(row.strip())
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train):
			fi = open(self.path_train, "r")
			rows = fi.readlines()
			for row in rows:
				row.strip()
				row=row.split(',')
				self.documents_train.append(row[1].strip('"')+' '+row[2].strip('"'))
				self.labels_train.append(classes[int(row[0])])
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test):
			fi = open(self.path_test, "r")
			rows = fi.readlines()
			for row in rows:
				row.strip()
				row=row.split(',')
				self.documents_test.append(row[1].strip('"')+' '+row[2].strip('"'))
				self.labels_test.append(classes[int(row[0])])
		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description):
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
		
	def load_agnews(self):
		self.documents_train,self.labels_train = [],[]
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir('agnews')
		self.path_dir = os.path.join(self.path,'agnews')
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

		self.path_train = os.path.join(self.path_dir,'train.csv')
		if not os.path.isfile(self.path_train):
			if self.verbose:
				print('Downloading agnews...')
			url='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
			wget.download(url, out=self.path_train)
			'''
			with bz2.open('temp.bz2', "rb") as f:
				# Decompress data from file
				content = f.read()
			with open(self.path_train, "w") as fichier:
				fichier.write(str(content))
			os.remove('temp.bz2')
			'''
			if self.verbose:
				print("\tCompleted!")
		if os.path.isfile(self.path_train):
			path_file=os.path.join(self.path_dir,'train.csv')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split('","')
				self.documents_train.append(row[1]+row[2])
				self.labels_train.append(row[0][1:])
			
	def load_trec(self):
		self.path_dir = os.path.join(self.path,'trec')
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

		self.path_train = os.path.join(self.path_dir,'train_5500.label')
		self.path_test  = os.path.join(self.path_dir,'TREC_10.label')
		if not os.path.isfile(self.path_train) or not os.path.isfile(self.path_test):
			if self.verbose:
				print('Downloading trec...')
			url='https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
			wget.download(url, out=self.path_train)
			url='https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
			wget.download(url, out=self.path_test)
			if self.verbose:
				print("\tCompleted!")
		if os.path.isfile(self.path_train):
			self.documents_train=[]
			self.labels_train=[]
			fi = open(self.path_train, encoding="ISO-8859-1")
			rows = fi.readlines()
			for row in rows:
				row=row.split(':')
				self.documents_train.append(row[1])
				self.labels_train.append(row[0])
				
		if os.path.isfile(self.path_test):
			self.documents_test=[]
			self.labels_test=[]
			fi = open(self.path_train, encoding="ISO-8859-1")
			rows = fi.readlines()
			for row in rows:
				row=row.split(':')
				self.documents_test.append(row[1])
				self.labels_test.append(row[0])

	def load_yelp(self):
		self.path_dir = os.path.join(self.path,'yelp')
		if not os.path.isdir(self.path_dir):
			os.mkdir(self.path_dir)

		self.path_train = os.path.join(self.path_dir,'kkk')
		self.path_test  = os.path.join(self.path_dir,'kk')
		
		if not os.path.isfile(self.path_train) or not os.path.isfile(self.path_test):
			if self.verbose:
				print('Downloading Yelp...')
			url='https://www.kaggle.com/yelp-dataset/yelp-dataset/download/yelp-dataset.zip'
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")
				
	def load_drugscom(self):
		"""
		Example pubmed_rct20k::

			from Manteia.Dataset import Dataset

			ds=Dataset('drugscom')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.path_dir = os.path.join(self.path,'drugscom')
		if not os.path.isdir(self.path_dir):
			os.mkdir(path_dir)

		self.path_train = os.path.join(self.path_dir,'drugsComTrain_raw.tsv')
		self.path_test  = os.path.join(self.path_dir,'drugsComTest_raw.tsv')
		
		if not os.path.isfile(self.path_train) or not os.path.isfile(self.path_test):
			if self.verbose:
				print('Downloading Drugs.com...')
			url='https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")
				
		if os.path.isfile(self.path_train):
			self.documents_train=[]
			self.labels_train=[]
			fi = open(self.path_train)
			reader = csv.DictReader(fi, delimiter = '\t')
			for row in reader:
				self.documents_train.append(row['review'])
				self.labels_train.append(row['rating'])
		if os.path.isfile(self.path_test):
			self.documents_test=[]
			self.labels_test=[]
			fi = open(self.path_test)
			reader = csv.DictReader(fi, delimiter = '\t')
			for row in reader:
				self.documents_test.append(row['review'])
				self.labels_test.append(row['rating'])

	def load_SST_2(self):
		file_train=self.path+'/SST-2/train.tsv'
		file_dev=self.path+'/SST-2/dev.tsv'
		file_test=self.path+'/SST-2/test.tsv'
		if not os.path.isfile(file_train) and not os.path.isfile(file_test):
			url='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
			if self.verbose:
				print("Downloading and extracting SST-2...")
			download_and_extract(url,'./dataset')
			if self.verbose:
				print("\tCompleted!")
		else:
			df_train = pd.read_csv(file_train,sep = "	", header = 0,names=['sentence','label'])
			df_dev = pd.read_csv(file_dev,sep = "	", header = 0,names=['sentence','label'])
			df_test = pd.read_csv(file_test,sep = "	", header = 0,names=['sentence'])
			if self.verbose==True:
				print(df_train.head())
				print(df_dev.head())
				print(df_test.head())
				
			self.documents_train = df_train['sentence'].values
			self.labels_train    = df_train['label'].values

			self.documents_dev   = df_dev['sentence'].values
			self.labels_dev      = df_dev['label'].values

			self.documents_test  = df_test['sentence'].values
			self.labels_test     = df_test['label'].values


	def del_dir(self,name):
		clear_folder(name)
			
	def load_SST_B(self):
		file_train = self.path+'/STS-B/train.tsv'
		file_dev   = self.path+'/STS-B/dev.tsv'
		file_test  = self.path+'/STS-B/test.tsv'
		if not os.path.isfile(file_train) and not os.path.isfile(file_test):
			url='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5'
			if self.verbose:
				print("Downloading and extracting SST-B...")
			download_and_extract(url,'./dataset')
			if self.verbose:
				print("\tCompleted!")
		else:
			if self.verbose:
				print("Loading and extracting SST-B...")
			df_train = pd.read_csv(file_train,sep = "	", header = 0,names=['sentence','label'])
			df_dev   = pd.read_csv(file_dev,sep = "	", header = 0,names=['sentence','label'])
			df_test  = pd.read_csv(file_test,sep = "	", header = 0,names=['sentence'])
			if self.verbose==True:
				print(df_train.head())
				print(df_dev.head())
				print(df_test.head())
				
			self.documents_train = df_train['sentence'].values
			self.labels_train    = df_train['label'].values

			self.documents_dev   = df_dev['sentence'].values
			self.labels_dev      = df_dev['label'].values

			self.documents_test  = df_test['sentence'].values
			self.labels_test     = df_test['label'].values

	def load_pubmed_rct20k(self):
		"""
		Example pubmed_rct20k::

			from Manteia.Dataset import Dataset

			ds=Dataset('pubmed_rct20k')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test   = [],[]
		self.documents_dev,self.labels_dev     = [],[]
		
		path_dir=os.path.join(self.path,'PubMed_20k_RCT')
		if not os.path.isdir(path_dir):
			os.mkdir(path_dir)
		
			url_train = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/train.txt'
			url_dev   = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/dev.txt'
			url_test  = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/test.txt'
			if self.verbose:
				print("Downloading and extracting pubmed-rct...")
			wget.download(url_train, out=path_dir)
			wget.download(url_test, out=path_dir)
			wget.download(url_dev, out=path_dir)
		path_file=os.path.join(path_dir,'train.txt')
		fi = open(path_file, "r")
		rows = fi.readlines()
		for row in rows:
			row=row.split('	')
			if len(row)==2:
				self.documents_train.append(row[1])
				self.labels_train.append(row[0])
		path_file=os.path.join(path_dir,'test.txt')
		fi = open(path_file, "r")
		rows = fi.readlines()
		for row in rows:
			row=row.split('	')
			if len(row)==2:
				self.documents_test.append(row[1])
				self.labels_test.append(row[0])
		path_file=os.path.join(path_dir,'dev.txt')
		fi = open(path_file, "r")
		rows = fi.readlines()
		for row in rows:
			row=row.split('	')
			if len(row)==2:
				self.documents_dev.append(row[1])
				self.labels_dev.append(row[0])



def download_and_extract(url, data_dir):
		data_file = "temp.zip"
		if os.path.isfile(data_file):
			os.remove(data_file)

		#urllib.request.urlretrieve(url, data_file)# test le remplacement!!!!
		
		wget.download(url, out=data_file,bar=bar_progress)
		
		with zipfile.ZipFile(data_file) as zip_ref:
			zip_ref.extractall(data_dir)
		#clean
		if os.path.isfile(data_file):
			os.remove(data_file)
"""
del directorie and is content.
"""
def clear_folder(dir):
    print('clear : '+dir)
    if os.path.exists(dir):
        for the_file in os.listdir(dir):
            file_path = os.path.join(dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)
            except Exception as e:
                print(e)
        os.rmdir(dir)
        
def construct_sample(path_train,classes=None):
	documents_train,labels_train = [],[]
	if os.path.isfile(path_train):
			fi = open(path_train, "r")
			rows = fi.readlines()
			for row in rows:
				row.strip()
				row=row.split(',')
				if len(row)>2:
					documents_train.append(row[1].strip('"')+' '+row[2].strip('"'))
				else:
					documents_train.append(row[1].strip('"'))
				if classes is None:
					labels_train.append(row[0].strip('"'))
				else:
					labels_train.append(classes[int(row[0].strip('"'))])

	return documents_train,labels_train
	
def load_multiple_file(file_list,path,path_dir):
	for file_ in file_list:
			print('Downloading {}.'.format(file_))
			url='https://github.com/ym001/Dune/raw/master/datasets/'+file_
			path_file = os.path.join(path_dir,file_)
			wget.download(url, out=path_file,bar=bar_progress)
	data = None
	packet_size = int(74 * 1000**2)   # bytes
	for file_ in file_list:
				path_file = os.path.join(path_dir,file_)
				with open(path_file, "rb") as output:
					if data is None:data = output.read(packet_size)
					else:data += output.read(packet_size)
	path_file = os.path.join(path_dir,'temp.zip')
	with open(path_file, "wb") as packet:
				packet.write(data)
	with zipfile.ZipFile(path_file) as zip_ref:
				zip_ref.extractall(path)
	if os.path.isfile(path_file):
				os.remove(path_file)
	for file_ in file_list:
				path_file = os.path.join(path_dir,file_)
				os.remove(path_file)
