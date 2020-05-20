"""
.. module:: Visualisation
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
from .Preprocess import Preprocess
from .Statistic import Statistic
from .Model import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualisation:

	r"""
		This is the class to make visualisation of NLP task.
		
		Args:
		
				
			documents (:obj:`list`, optional, defaults to None):
				A list of documents.
				
			labels (:obj:`float`, optional, defaults to None):
				A list of labels.

			dataset_name (:obj:`string`, optional, defaults to ''):
				Name of the dataset.

			path (:obj:`string`, optional, defaults to ''):
				Path to save the report.

			save (:obj:`bool`, optional, defaults to False):
				save the graph to the path.

			show (:obj:`bool`, optional, defaults to False):
				show the graph.
				 
		Example::
		
			from Manteia.Statistic import Statistic 
			from Manteia.Visualisation import Visualisation
			
			documents = [
			'  !?? What do you call a potato in space? Spudnik:::13 ;;    //   ',
			'What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.',
			'What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',
			'Why do you never see elephants hiding in trees? \'Cause they are freaking good at it',
			'My son just got a tattoo of a heart, a spade, a club, and a diamond, all without my permission. I guess I\'ll deal with him later.',
			'Mom: "Do you want this?" Me: "No." Mom: "Ok I\'ll give it to your brother." Me: "No I want it."',
			'Ibuprofen is my favorite headache medicine that also sounds like a reggae professor.',
			'INTERVIEWER: Why do you want to work here? ME: *crumbs tumbling from my mouth* Oh, I don\'t. I was just walking by and saw you had donuts.',
			'I\'ve struggled for years to be above the influence... But I\'ve never been able to get that high',
			'With Facebook, you can stay in touch with people you would otherwise never talk to, but that\'s only one of the many awful things about it',
			]
			
			labels = [
			['funny'],['not funny'],['funny'],['not funny'],['funny'],['not funny'],['not funny'],['not funny'],['funny'],['not funny'],
			]
			
			stat=Statistic(documents,labels)
			dictionary=stat.dictionnary_stat_labels()
			path='./visu.png'
			visu = Visualisation(path)
			visu.format_data(dictionary)
			visu.plot_bar()
			
		Attributes:
	"""
	
	
	def __init__(self,path='',name='',save=False,show=True):
		self.path=path
		self.name=name
		self.save=save
		self.show=show
	def test(self):
		return "Visualisation Mantéïa."
	#Format de donnée en entrée un dictionnaire en sortie deux liste x,y 
	def format_data(self,dictionary=None):
		self.x=[]
		self.y=[]
		for key, value in dictionary.items():
			self.x.append(key)
			self.y.append(value)
			
	def plot_bar(self):
		#Plot the data:
		palette = plt.cm.get_cmap('tab10')

		plt.title(self.name)
		#Set tick colors:
		ax = plt.gca()
		ax.tick_params(axis='x', colors='gray')
		ax.tick_params(axis='y', colors='blue')
		
		plt.barh(self.x, self.y,color=palette.colors)
		if self.save:
			plt.savefig(self.path)
		if self.show:
			plt.show()

	def plot_boxplot(self,labels,longueur_des_doc,ylim=200):
		"""
		figure for show exploration length of document.
		"""
		data = pd.DataFrame({'Labels':labels ,'Length of document':longueur_des_doc})
		fig, ax = plt.subplots()
		plt.xticks(rotation=90) 
		sns.boxplot(x='Labels', y='Length of document', data=data, palette='Set2',notch=True,showfliers=True, showmeans=True, meanline=True)
		ax.set_ylim(0, ylim)
		plt.show()
		#plt.savefig('/home/mercadier/these/resultat/image/longueur-doc-by-classe.png')
		
		
