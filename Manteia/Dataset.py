#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  data.py
#  
#  Copyright 2017 yves <yves.mercadier@ac-montpellier.fr>
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
#  '

from .Preprocess import Preprocess
import numpy as np
from nltk.corpus import reuters,brown,webtext
from sklearn.datasets import fetch_20newsgroups

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

	def load_20newsgroups(self):
		#categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space']
		categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		doc=twenty_train.data
		label=[]
		for i in range(len(twenty_train.target)):
			label.append(categorie[twenty_train.target[i]])
		return doc,label
		
	def get_documents(self):
		return self.documents

	def get_labels(self):
		return self.labels
