
import unittest
from Manteia.Classification import Classification

class ClassificationTest(unittest.TestCase):

	"""Test case used to test the functions of the 'Manteia.Preprocess' module."""

	def test_Classification(self):
		print("Test the functioning of the class Manteia.Classification.")
		print("_________________________________________________________")
		cl=Classification()
        
		self.assertEqual(type(cl.test()), str)

	def test_Classification_init(self):
		print("Test the functioning of the class Manteia.Classification init.")
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
			
		cl=Classification(documents=documents,labels=labels,process=True)
        
		self.assertEqual(len(cl.predict(documents[:2])), 2)
        
unittest.main()
