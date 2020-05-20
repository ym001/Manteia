"""
.. module:: Statistic
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Yves Mercadier <manteia.ym001@gmail.com>


"""
from .Preprocess import Preprocess
import numpy as np

class Statistic:
	
	r"""
		This is the class to make statistic of text.
		
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
	def __init__(self,documents=[],labels=[],dataset_name='',path='',statistic=True,verbose=True):
		
		self.documents    = documents
		self.labels       = labels
		self.path         = path
		self.dataset_name = dataset_name
		self.verbose      = verbose
		
		if statistic==True and documents!=[] and labels!=[]:
			self.list_labels=self.list_labels(labels)
			if verbose:
				self.print_report()
		
	def test(self):
		return "Mantéïa Statistic."
		
	def type(self,labels=None):
		c_lab=0
		c_label=0
		for lab in labels:
			c_lab=c_lab+1
			for l in lab:
				c_label=c_label+1
		if c_lab==c_label:
			return "multiclasse"
		else:
			return "multilabel"
			
	def number_text(self):
		return len(self.documents)

	def dictionary(self,documents):
		dico = {}
		nb_word=0
		for doc in documents:
			tab=doc.split(" ")
			for word in tab:
				if word not in dico:
					dico[word]=1
				else:
					dico[word]=dico[word]+1
		return dico
		
	def number_word(self):
		return len(self.dictionary(self.documents))

		
	def dictionnary_stat_labels(self):
		"""
		récupére les stats de nombre de mot par label
		"""
		dic_word={}
		print(self.list_labels)
		for lab in self.list_labels:
			dic_word[lab]= 0
		for doc,lab in zip(self.documents,self.labels):
			tab=doc.split(" ")
			for l in lab:
				if l in self.list_labels:
					dic_word[l]=dic_word[l]+len(tab)
		return dic_word
		
	def length_of_documents_by_class(self):
		dic_length={}
		length_of_doc=[]
		classe=[]
		for lab in self.list_labels:
			dic_length[lab]= []
		for doc,cl in zip(self.documents,self.labels):
			sentence=doc.split(" ")
			length_of_doc.append(len(sentence))
			classe.append(cl)
			dic_length[cl].append(len(sentence))
		for lab in self.list_labels:
			tab=np.array(dic_length[lab])
			dic_length[lab]=np.mean(tab)

		'''a rajouter dans visualisation
		data = pd.DataFrame({'Classe':classe ,'Longueur des documents':longueur_des_doc})
		fig, ax = plt.subplots()
		plt.xticks(rotation=90) 
		sns.boxplot(x='Classe', y='Longueur des documents', data=data, palette='Set2',notch=True,showfliers=True, showmeans=True, meanline=True)
		ax.set_ylim(0, 200)
		plt.savefig('/home/mercadier/these/resultat/image/longueur-doc-by-classe.png')
		'''
		return dic_length
		
	def length_of_documents(self):
		length_of_doc=[]
		for doc in self.documents:
			sentence=doc.split(" ")
			length_of_doc.append(len(sentence))
			
		return length_of_doc
		
	def word_by_doc(self):
		c=0
		for doc in self.documents:
			tab=doc.split(" ")
			c=c+len(tab)
			
		return c/len(self.documents)
		
	def doc_classe(self,list_labels=None,documents=None,labels=None):
		dic_doc={}
		for lab in list_labels:
			dic_doc[lab]= 0
		for doc,lab in zip(documents,labels):
			dic_doc[lab]=dic_doc[lab]+1
		return dic_doc
		
	def len_doc(self):
		len_doc=0
		for doc in self.documents:
			tab=doc.split(" ")
			len_doc=len(tab)
		return 	len_doc/len(self.documents)
		
	'''idem
	def class_imbalance(self):
		classe=[]
		height=[]
		for key, value in self.dic_doc.items():
			classe.append(key)
			height.append(value)
	'''
		
	def report(self):
		report=''
		report+="Dataset : {}\n".format(self.dataset_name)
		report+="Number of documents : {}\n".format(self.number_text())
		report+="Type : {}\n".format(self.type(self.labels))
		report+="List of labels : {}\n".format(self.list_labels)
		report+="Number of classes : {}\n".format(len(self.list_labels))
		report+="Word count per document : {}\n".format(self.word_by_doc())
		report+="Unique word count : {}\n".format(self.number_word())
		report+="Document count per classe : {}\n".format(self.doc_classe(self.list_labels,self.documents,self.labels))
		report+="\n"
		return report

		
	def print_report(self):
		print(self.report())
		
	def save_report(self):
		fichier=self.path+"statistical_report_"+self.dataset_name+".txt"
		mon_fichier = open(fichier, "w") 
		mon_fichier.write(self.report)
		mon_fichier.close()
		
	def list_labels(self,labels):
		return np.sort(np.unique(np.array(labels)), axis=0)
