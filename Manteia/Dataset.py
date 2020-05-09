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
	def __init__(self,name='20newsgroups'):

		self.name=name
		self.load()
		
	def test(self):
		return "Mantéïa Dataset."

	def load(self):
		if self.name=="20newsgroups":
			self.documents,self.labels=self.load_20newsgroups()
		if self.name=="SST-2":
			self.documents,self.labels=self.load_SST_2()
		if self.name=="SST-B":
			self.documents,self.labels=self.load_SST_B()

	def load_20newsgroups(self):
		#categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space']
		categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		doc=twenty_train.data
		label=[]
		for i in range(len(twenty_train.target)):
			label.append(categorie[twenty_train.target[i]])
		return doc,label

	def load_SST_2(self):
		url='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
		print("Downloading and extracting SST-2...")
		self.download_and_extract(url,'./dataset')

	def load_SST_B(self):
		url='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5'
		print("Downloading and extracting SST-B...")
		self.download_and_extract(url,'./dataset')

	def download_and_extract(url, data_dir):
		data_file = "temp"
		urllib.request.urlretrieve(url, data_file)
		with zipfile.ZipFile(data_file) as zip_ref:
			zip_ref.extractall(data_dir)
		os.remove(data_file)
		print("\tCompleted!")
		
	def get_documents(self):
		return self.documents

	def get_labels(self):
		return self.labels
