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
		This is the class description in order to get some dataset.
		
		
		* **name**        - name of the dataset (str)
		* **train**       - load the dataset train Default: ‘True’.
		* **test**        - load the dataset test Default: ‘False’.
		* **dev**         - load the dataset dev Default: ‘False’.
		* **description** - load description Default: ‘False’.
		* **verbose**     - produce and display some explanation
		* **path**        - Path to the data file.
		
	"""
	def __init__(self,name='20newsgroups',train=True,test=False,dev=False,classe=True,desc=False,path='./dataset',verbose=True):
		r"""
		"""
		self.name=name
		self.train=train
		self.test=test
		self.dev=dev
		self.classe=classe
		self.desc=desc
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
		if self.name=="SST-5":
			self.load_SST_5()
			

		if self.name=="COVID":
			self.load_COVID()
			
		if self.name=="drugscom":
			self.load_drugscom()
			
		if self.name=="pubmed_rct20k":
			self.load_pubmed_rct20k()

		if self.name=="eRisk_anx":
			self.load_eRisk_anx()
		if self.name=="eRisk_dep":
			self.load_eRisk_dep()
			
		if self.name=="yelp":
			self.load_yelp()
			
		if self.name=="trec":
			self.load_trec()
			
		if self.name=="Agnews":
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

		if self.name=="Short_Jokes":
			self.load_Short_Jokes()

		if self.name=="Tweeter Airline Sentiment":
			self.load_Tweeter_Airline_Sentiment()

			
	def load_20newsgroups(self):
		r"""
		Defines 20newsgroups datasets.
			The labels includes:
			
			* 0 : sci.crypt.
			* 1 : sci.electronics.
			* 2 : sci.med.
			* 3 : sci.space.
			* 4 : rec.autos.
			* 5 : rec.sport.baseball.
			* 6 : rec.sport.hockey.
			* 7 : talk.politics.guns.
			* 8 : talk.politics.mideast.
			* 9 : talk.politics.misc.
			* 10 : talk.religion.misc.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('20newsgroups')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		
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
		"""
		Defines Yelp Review Polarity datasets.
			The labels includes:
			
			* 1 : Negative polarity.

			* 2 : Positive polarity.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Yelp Review Polarity',test=True,desc=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])
			print(ds.description)
		"""
		self.path_dir = os.path.join(self.path,'yelp_review_polarity')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

			file_list=['yelp_review_polarity00.zip','yelp_review_polarity01.zip','yelp_review_polarity02.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train) and self.train:
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test) and self.test:
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description) and self.desc:
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
				
	def load_Yelp_Review_Full(self):
		r"""
		Defines Yelp Review Full Star Dataset.
			The labels includes:
			
			**1 - 5** : rating classes (5 is highly recommended).

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Yelp Review Full',test=True,desc=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])

			print('Test : ')
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])

			print('Description :')
			print(ds.description)
		"""
		self.path_dir = os.path.join(self.path,'yelp_review_full')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

			file_list=['yelp_review_full00.zip','yelp_review_full01.zip','yelp_review_full02.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train) and self.train:
			self.documents_train,self.labels_train=construct_sample(self.path_train)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test)and self.test:
			self.documents_test,self.labels_test=construct_sample(self.path_test)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description)and self.desc:
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row

	def load_Yahoo_Answers(self):
		r"""
		Defines Yahoo! Answers datasets.
			The labels includes:
			
			* Society & Culture
			* Science & Mathematics
			* Health
			* Education & Reference
			* Computers & Internet
			* Sports
			* Business & Finance
			* Entertainment & Music
			* Family & Relationships
			* Politics & Government

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Yahoo! Answers',test=True,desc=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])

			print('Test : ')
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])

			print('Description :')
			print(ds.description)

			print('List labels :')
			print(ds.list_labels)
		"""
		self.path_dir = os.path.join(self.path,'yahoo_answers')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

			file_list=['yahoo_answers00.zip','yahoo_answers01.zip','yahoo_answers02.zip','yahoo_answers03.zip',
						'yahoo_answers04.zip']
			load_multiple_file(file_list,self.path,self.path_dir)
			
		self.path_classes = os.path.join(self.path_dir,'classes.txt')
		classes=['']
		if os.path.isfile(self.path_classes)and classes:
			fi = open(self.path_classes, "r")
			rows = fi.readlines()
			for row in rows:
				classes.append(row.strip())
			self.list_labels=classes

				
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train)and self.train:
			self.documents_train,self.labels_train=construct_sample(self.path_train,classes)
		
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test)and self.test:
			self.documents_test,self.labels_test=construct_sample(self.path_test,classes)

		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description)and self.desc:
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
	def load_Sogou_News(self):
		
		self.path_dir = os.path.join(self.path,'sogou_news')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

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

	
	def load_Amazon_Review_Polarity(self):
		"""
		Defines Amazon Review Polarity datasets.
			The labels includes:
			
			* 1 : Negative polarity.

			* 2 : Positive polarity.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Amazon Review Polarity',test=True,desc=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])
			print(ds.description)
		"""
		self.path_dir = os.path.join(self.path,'amazon_review_polarity')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

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
		r"""
		Defines Amazon Review Full Star Dataset.
			The labels includes:
			
			**1 - 5** : rating classes (5 is highly recommended).

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Amazon Review Full',test=True,desc=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])

			print('Test : ')
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])

			print('Description :')
			print(ds.description)
		"""
		self.path_dir = os.path.join(self.path,'amazon_review_full')
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!

		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

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
		r"""
		Defines DBPedia datasets.
			The labels includes:
			
			* Company
			* EducationalInstitution
			* Artist
			* Athlete
			* OfficeHolder
			* MeanOfTransportation
			* Building
			* NaturalPlace
			* Village
			* Animal
			* Plant
			* Album
			* Film
			* WrittenWork

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('DBPedia',test=True,desc=True,classe=True)

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])

			print('Test : ')
			print(ds.documents_test[:5])
			print(ds.labels_test[:5])

			print('Description :')
			print(ds.description)

			print('List labels :')
			print(ds.list_labels)
		"""
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test = [],[]
		self.path_dir = os.path.join(self.path,'DBPedia')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!

		if not os.path.isdir(self.path_dir):
			
			if self.verbose:
				print('Downloading DBPedia.')
			url='https://github.com/ym001/Dune/raw/master/datasets/DBPedia.zip'
			download_and_extract(url, self.path)
			
		self.path_classes = os.path.join(self.path_dir,'classes.txt')
		classes=[]
		if os.path.isfile(self.path_classes) and self.classe:
			fi = open(self.path_classes, "r")
			rows = fi.readlines()
			for row in rows:
					classes.append(row.strip())
			self.list_labels=classes
		self.path_train = os.path.join(self.path_dir,'train.csv')
		if os.path.isfile(self.path_train)and self.train:
			fi = open(self.path_train, "r")
			rows = fi.readlines()
			for row in rows:
				row.strip()
				row=row.split(',')
				
				self.documents_train.append(row[1].strip('"')+' '+row[2].strip('"'))
				self.labels_train.append(classes[int(row[0])-1])
		self.path_test = os.path.join(self.path_dir,'test.csv')
		if os.path.isfile(self.path_test)and self.test:
			fi = open(self.path_test, "r")
			rows = fi.readlines()
			for row in rows:
				row.strip()
				row=row.split(',')
				self.documents_test.append(row[1].strip('"')+' '+row[2].strip('"'))
				self.labels_test.append(classes[int(row[0])-1])
		self.path_description = os.path.join(self.path_dir,'readme.txt')
		if os.path.isfile(self.path_description)and self.desc:
			self.description=''
			fi = open(self.path_description, "r")
			rows = fi.readlines()
			for row in rows:
				self.description+=row
		
	def load_agnews(self):
		r"""
		Defines Agnews datasets.
			The labels includes:
			
			* 0 : World
			* 1 : Sports
			* 2 : Business
			* 3 : Sci/Tech

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('agnews')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train,self.labels_train = [],[]
		#!!!!!!!!!!!!!!!!!!!!
		self.del_dir('agnews')
		self.path_dir = os.path.join(self.path,'agnews')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

		self.path_train = os.path.join(self.path_dir,'train.csv')
		classes=['World','Sports','Business','Sci/Tech']
		if self.classe:
			self.list_labels=classes
		if not os.path.isfile(self.path_train):
			if self.verbose:
				print('Downloading Agnews...')
			url='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
			wget.download(url, out=self.path_train)
			if self.verbose:
				print("\tCompleted!")
		if os.path.isfile(self.path_train) and self.train:
			path_file=os.path.join(self.path_dir,'train.csv')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split('","')
				self.documents_train.append(row[1]+row[2])
				self.labels_train.append(classes[int(row[0][1:])-1])
		
			
	def load_trec(self):
		r"""
		Defines Trec datasets.
			The labels includes:
			
			* ABBREVIATION
			* ENTITY
			* DESCRIPTION
			* HUMAN
			* LOCATION
			* NUMERIC

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('agnews')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.path_dir = os.path.join(self.path,'trec')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

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
		if os.path.isfile(self.path_train) and self.train:
			self.documents_train=[]
			self.labels_train=[]
			fi = open(self.path_train, encoding="ISO-8859-1")
			rows = fi.readlines()
			for row in rows:
				row=row.split(':')
				self.documents_train.append(row[1])
				self.labels_train.append(row[0])
				
		if os.path.isfile(self.path_test)and self.train:
			self.documents_test=[]
			self.labels_test=[]
			fi = open(self.path_train, encoding="ISO-8859-1")
			rows = fi.readlines()
			for row in rows:
				row=row.split(':')
				self.documents_test.append(row[1])
				self.labels_test.append(row[0])

	def load_drugscom(self):
		r"""
		Defines Drugs.com Dataset.
			The labels includes:
			
			**0 - 9** : rating classes (9 is highly).

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('drugscom')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.path_dir = os.path.join(self.path,'drugscom')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)

		self.path_train = os.path.join(self.path_dir,'drugsComTrain_raw.tsv')
		self.path_test  = os.path.join(self.path_dir,'drugsComTest_raw.tsv')
		
		if not os.path.isfile(self.path_train) or not os.path.isfile(self.path_test):
			if self.verbose:
				print('Downloading Drugs.com...')
			url='https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")
				
		if os.path.isfile(self.path_train)and self.train:
			self.documents_train=[]
			self.labels_train=[]
			fi = open(self.path_train)
			reader = csv.DictReader(fi, delimiter = '\t')
			for row in reader:
				self.documents_train.append(row['review'])
				self.labels_train.append(row['rating'])
		if os.path.isfile(self.path_test)and self.test:
			self.documents_test=[]
			self.labels_test=[]
			fi = open(self.path_test)
			reader = csv.DictReader(fi, delimiter = '\t')
			for row in reader:
				self.documents_test.append(row['review'])
				self.labels_test.append(row['rating'])

	def load_eRisk_anx(self):
		
		self.path_dir = os.path.join(self.path,'eRisk_anx')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
		
		if not os.path.isdir(os.path.join(self.path_dir,'pos')):
			if self.verbose:
				print('Downloading eRisk_anx...')
			url='https://github.com/ym001/Dune/raw/master/datasets/eRisk_anx.zip'
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")
				
		if os.path.isdir(os.path.join(self.path_dir,'neg'))and self.train:
			self.documents_train=[]
			self.labels_train=[]
			dossier=os.path.join(self.path_dir,'neg')
			FichList = [ f for f in os.listdir(dossier)]
			for fich in FichList:
				with open(os.path.join(dossier,fich),'r') as f:
					text = f.read()
				self.documents_train.append(text)
				self.labels_train.append('-')
			dossier=os.path.join(self.path_dir,'pos')
			FichList = [ f for f in os.listdir(dossier)]
			for fich in FichList:
				with open(os.path.join(dossier,fich),'r') as f:
					text = f.read()
				self.documents_train.append(text)
				self.labels_train.append('+')

	def load_eRisk_dep(self):
		
		self.path_dir = os.path.join(self.path,'eRisk_dep')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
			file_list=['eRisk_dep00.zip','eRisk_dep01.zip','eRisk_dep02.zip']
			load_multiple_file(file_list,self.path_dir,self.path_dir)
				
		if os.path.isdir(os.path.join(self.path_dir,'neg'))and self.train:
			self.documents_train=[]
			self.labels_train=[]
			dossier=os.path.join(self.path_dir,'neg')
			FichList = [ f for f in os.listdir(dossier)]
			for fich in FichList:
				with open(os.path.join(dossier,fich),'r') as f:
					text = f.read()
				self.documents_train.append(text)
				self.labels_train.append('-')
			dossier=os.path.join(self.path_dir,'pos')
			FichList = [ f for f in os.listdir(dossier)]
			for fich in FichList:
				with open(os.path.join(dossier,fich),'r') as f:
					text = f.read()
				self.documents_train.append(text)
				self.labels_train.append('+')

	def load_COVID(self):
		
		self.path_dir = os.path.join(self.path,'COVID')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
		self.path_train = os.path.join(self.path_dir,'covid.zip')
		
		if not os.path.isfile(self.path_train):
			if self.verbose:
				print('Downloading COVID...')
			url='https://github.com/ym001/Dune/raw/master/datasets/covid.zip'
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")
				
		if os.path.isfile(os.path.join(self.path_dir,'all_sources_metadata_2020-03-13.csv'))and self.train:
			self.documents_train=[]
			self.labels_train=[]
			file_train=os.path.join(self.path_dir,'all_sources_metadata_2020-03-13.csv')

			fh = open(file_train)
			reader = csv.DictReader(fh, delimiter = ',')
			for ligne in reader:
				if ligne['abstract']!='':
					self.documents_train.append(ligne['title']+' '+ligne['abstract'])
					self.labels_train.append(ligne['source_x'])
			fh.close()

	def load_SST_2(self):
		"""
		Defines SST 2 datasets.
			The labels includes:
			
			* Negative polarity.

			* Positive polarity.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('SST-2')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""

		self.path_dir = os.path.join(self.path,'stanfordSentimentTreebank')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
			print('dossier')
		file_sentiment = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_sentiment = os.path.join(file_sentiment,'sentiment_labels.txt')

		if not os.path.isfile(file_sentiment):
			url='http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'

			if self.verbose:
				print("Downloading and extracting SST-2...")
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")

		sentiments = {}
		fi = open(file_sentiment, "r")
		rows = fi.readlines()
		for row in rows:
				row=row.split('|')
				if len(row)==2:
					sentiments[row[0]]=row[1].strip()
		file_Sentences = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_Sentences = os.path.join(file_Sentences,'datasetSentences.txt')
		sentences= {}
		fi = open(file_Sentences, "r")
		rows = fi.readlines()
		for i,row in enumerate(rows):
			if i>0:
				row=row.split('	')
				sentences[row[0]]=row[1].strip()
					
		file_Split = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_Split = os.path.join(file_Split,'datasetSplit.txt')
		ids_train,ids_test,ids_dev = [],[],[]
		fi = open(file_Split, "r")
		rows = fi.readlines()
		for row in rows:
				row=row.strip()
				row=row.split(',')
				if row[1]=='1':
						ids_train.append(row[0])
				if row[1]=='2':
						ids_test.append(row[0])
				if row[1]=='3':
						ids_dev.append(row[0])
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test = [],[]
		self.documents_dev,self.labels_dev = [],[]

		for ids in sentences.keys():
				sentence=sentences[ids]
				sentiment=float(sentiments[ids])
				if sentiment < 0.5:
					label = 'negative'
				else:
					label = 'positive'
				if self.train and ids in ids_train: 
					self.documents_train.append(sentence)
					self.labels_train.append(label)
				if self.test and ids in ids_test: 
					self.documents_test.append(sentence)
					self.labels_test.append(label)
				if self.dev and ids in ids_dev: 
					self.documents_dev.append(sentence)
					self.labels_dev.append(label)

	def load_SST_5(self):
		"""
		Defines SST 5 datasets.
			The labels includes:
			
			* very negative.
			* negative.
			* neutral.
			* positive.
			* very positive.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('SST-5',dev=True)

			print('Dev : ')
			print(ds.documents_dev[:5])
			print(ds.labels_dev[:5])
		"""

		self.path_dir = os.path.join(self.path,'stanfordSentimentTreebank')
		#!!!!!!!!!!!!!!!!!!!!
		#self.del_dir(self.path_dir)
		#!!!!!!!!!!!!!!!!!!!!
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
			print('dossier')
		file_sentiment = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_sentiment = os.path.join(file_sentiment,'sentiment_labels.txt')

		if not os.path.isfile(file_sentiment):
			url='http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'

			if self.verbose:
				print("Downloading and extracting SST-5...")
			download_and_extract(url,self.path_dir)
			if self.verbose:
				print("\tCompleted!")

		sentiments = {}
		fi = open(file_sentiment, "r")
		rows = fi.readlines()
		for row in rows:
				row=row.split('|')
				if len(row)==2:
					sentiments[row[0]]=row[1].strip()
		file_Sentences = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_Sentences = os.path.join(file_Sentences,'datasetSentences.txt')
		sentences= {}
		fi = open(file_Sentences, "r")
		rows = fi.readlines()
		for i,row in enumerate(rows):
			if i>0:
				row=row.split('	')
				sentences[row[0]]=row[1].strip()
					
		file_Split = os.path.join(self.path_dir,'stanfordSentimentTreebank')
		file_Split = os.path.join(file_Split,'datasetSplit.txt')
		ids_train,ids_test,ids_dev = [],[],[]
		fi = open(file_Split, "r")
		rows = fi.readlines()
		for row in rows:
				row=row.strip()
				row=row.split(',')
				if row[1]=='1':
						ids_train.append(row[0])
				if row[1]=='2':
						ids_test.append(row[0])
				if row[1]=='3':
						ids_dev.append(row[0])
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test = [],[]
		self.documents_dev,self.labels_dev = [],[]
		for ids in sentences.keys():
				sentence=sentences[ids]
				sentiment=float(sentiments[ids])
				if sentiment <= 0.2:
					label = 'very negative'
				if sentiment > 0.2 and sentiment <= 0.4:
					label = 'negative'
				if sentiment > 0.4 and sentiment <= 0.6:
					label = 'neutral'
				if sentiment > 0.6 and sentiment <= 0.8:
					label = 'positive'
				if sentiment > 0.8:
					label = 'very positive'
				if self.train and ids in ids_train: 
					self.documents_train.append(sentence)
					self.labels_train.append(label)
				if self.test and ids in ids_test: 
					self.documents_test.append(sentence)
					self.labels_test.append(label)
				if self.dev and ids in ids_dev: 
					self.documents_dev.append(sentence)
					self.labels_dev.append(label)

			
	def del_dir(self,name):
		"""
		Delete file of the dataset.
		"""
		clear_folder(name)

	def load_pubmed_rct20k(self):

		r"""
		Defines Pubmed RCT20k datasets.
			The labels includes:
			
			* BACKGROUND.
			* CONCLUSIONS.
			* METHODS.
			* OBJECTIVE.
			* RESULTS.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('pubmed_rct20k')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train,self.labels_train = [],[]
		self.documents_test,self.labels_test   = [],[]
		self.documents_dev,self.labels_dev     = [],[]
		
		self.path_dir=os.path.join(self.path,'PubMed_20k_RCT')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
		
			url_train = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/train.txt'
			url_dev   = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/dev.txt'
			url_test  = 'https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/test.txt'
			if self.verbose:
				print("Downloading and extracting pubmed-rct...")
			wget.download(url_train, out=path_dir)
			wget.download(url_test, out=path_dir)
			wget.download(url_dev, out=path_dir)
		if self.train:
			path_file=os.path.join(self.path_dir,'train.txt')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split('	')
				if len(row)==2:
					self.documents_train.append(row[1])
					self.labels_train.append(row[0])
		if self.test:
			path_file=os.path.join(path_dir,'test.txt')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split('	')
				if len(row)==2:
					self.documents_test.append(row[1])
					self.labels_test.append(row[0])
		if self.dev:
			path_file=os.path.join(path_dir,'dev.txt')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split('	')
				if len(row)==2:
					self.documents_dev.append(row[1])
					self.labels_dev.append(row[0])

	def load_Short_Jokes(self):

		r"""
		Defines Short_Jokes dataset.
			

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('pubmed_rct20k')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train = []
		
		self.path_dir=os.path.join(self.path,'Short_Jokes')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
			url_train = 'https://github.com/ym001/Dune/raw/master/datasets/short-jokes.zip'
			if self.verbose:
				print("Downloading and extracting Short_Jokes...")
			download_and_extract(url_train, self.path_dir)
		if self.train:
			path_file=os.path.join(self.path_dir,'shortjokes.csv')
			fi = open(path_file, "r")
			rows = fi.readlines()
			for row in rows:
				row=row.split(',')
				if len(row)==2:
					self.documents_train.append(row[1].strip())


	def load_Tweeter_Airline_Sentiment(self):

		r"""
		Defines Tweeter Airline Sentiment dataset.
			The labels includes:
			
			* positive.
			* neutral.
			* negative.

		.. code-block:: python

			from Manteia.Dataset import Dataset

			ds=Dataset('Tweeter Airline Sentiment')

			print('Train : ')
			print(ds.documents_train[:5])
			print(ds.labels_train[:5])
		"""
		self.documents_train = []
		self.labels_train = []
		
		self.path_dir=os.path.join(self.path,'Tweeter_Airline_Sentiment')
		if not os.path.isdir(self.path_dir):
			os.makedirs(self.path_dir)
			url_train = 'https://github.com/ym001/Dune/raw/master/datasets/Airline-Sentiment.zip'
			if self.verbose:
				print("Downloading and extracting Tweeter_Airline_Sentiment...")
			download_and_extract(url_train, path_dir)
		if self.train:
			path_file=os.path.join(path_dir,'Airline-Sentiment.csv')
			fi = open(path_file, "r")
			reader = csv.DictReader(fi, delimiter = ',')
			for row in reader:
				self.documents_train.append(row['text'])
				self.labels_train.append(row['airline_sentiment'])

def download_and_extract(url, data_dir):
		"""
		download_and_extract file of dataset.
		"""
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

def clear_folder(dir):
	"""
	Del directorie and is content.
	"""
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
				rint(e)
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
