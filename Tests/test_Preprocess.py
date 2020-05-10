
import unittest
from Manteia.Preprocess import Preprocess

class PreprocessTest(unittest.TestCase):

	"""Test case used to test the functions of the 'Manteia.Preprocess' module."""

	def test_Preprocess(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		print("_____________________________________________________")
		pp=Preprocess()
        
		# Vérifie que 'mant' est une 'str'
		self.assertEqual(type(pp.test()), str)

	def test_Preprocess_init(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		print("_____________________________________________________")
		pp=Preprocess(documents=['a','b'],labels=['a','b'])
        
		self.assertEqual(len(pp.documents), 2)
        
unittest.main()
