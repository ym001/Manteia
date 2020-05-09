
"""
	This module proclaims the good word. May they
	regain total freedom of artificial thought towards a new age
	reminiscent.

	You can install it with pip:

		pip install Manteia

	Example of use:

	>>> from Manteia import testManteia
	>>> testManteia ()

	This code is licensed under MIT.
"""
__all__ = ['testManteia','testData','testClassification']

from .Preprocess import Preprocess
from .Classification import Classification
from .Statistic import Statistic
from .Visualisation import Visualisation
from .Model import Model

class Task:
	def __init__(self,documents=None,labels=None,task='classification'):
		if documents!=None:
			self.data=Data(documents,labels)
		if task=='classification' and documents!=None:
			self.classification=Classification(data=self.data)
	def test(self):
		return "Hello, Task Mantéïa is alive."
		
def testManteia():
    print ("Hello, Mantéïa is alive.")

def testData():
	documents=['    ,;:123test   car','test houses']
	labels=['1','0']
	mant=Data(documents,labels)
	print(mant.data.list_labels)
	print(mant.data.get_df())

def testClassification():
	documents=['test car','test house']
	labels=['1','0']
	mant=Classification(documents,labels)

