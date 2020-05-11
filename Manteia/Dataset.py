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
import wget

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

class Dataset:
	
	r"""
		This is the class to give datasets.
		
		Args:

			dataset_name (:obj:`string`, optional, defaults to ''):
				Name of the dataset.
				 
		Example::

			from Manteia.Dataset import Dataset

			ds=Dataset('20newsgroups')
			documents=ds.get_documents()
			labels=ds.get_labels()

			print(documents[:5])
			print(labels[:5])

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
			self.documents,self.labels=self.load_20newsgroups()
		if self.name=="SST-2":
			self.load_SST_2()
		if self.name=="SST-B":
			self.load_SST_B()
		if self.name=="pubmed_rct20k":
			self.load_pubmed_rct20k()

	def load_20newsgroups(self):
		if self.verbose:
			print('Downloading 20newsgroups...')
		#categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space']
		categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		doc=twenty_train.data
		label=[]
		for i in range(len(twenty_train.target)):
			label.append(categorie[twenty_train.target[i]])
		return doc,label

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


	def del_file(self,name):
		if self.verbose:
			print('delete file : '+self.name)
		if self.name=='SST-2':
			file_dir=self.path+'/SST-2/'
			clear_folder(file_dir)
			
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
		data_file = "temp"
		urllib.request.urlretrieve(url, data_file)
		with zipfile.ZipFile(data_file) as zip_ref:
			zip_ref.extractall(data_dir)
		os.remove(data_file)

def clear_folder(dir):
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
