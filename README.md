Manteia - proclaim the good word
================================================================

Designing your neural network to natural language processing. Deep learning has been used extensively in natural language processing (NLP) because
it is well suited for learning the complex underlying structure of a sentence and semantic proximity of various words.
Data cleaning, construction model (Bert, Roberta, Distilbert, XLNet, Albert, GPT, GPT2),
quality measurement training and finally visualization of your results on several dataset ( 20newsgroups, SST-2, PubMed_20k_RCT, DBPedia, Amazon Review Full, Amazon Review Polarity)..


You can install it with pip :

     __pip install Manteia__

Example of use Classification :


	from Manteia.Classification import Classification
	documents=['a text','text b']  
	labels=['a','b']'  
	Classification(model_name ='roberta',documents,labels,process=True)


Example of use Generation :


	from Manteia.Generation import Generation
	Generation(seed='What do you do if a bird shits on your car?')
	If you're a car owner, you're supposed to be able to call the police and have them take the bird off the car.

[Documentation](https://manteia.readthedocs.io/en/latest/#)
[Pypi](https://pypi.org/project/Manteia/)
[Source](https://github.com/ym001/Manteia)

This code is licensed under MIT.
