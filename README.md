Manteia - proclaim the good word
================================================================

Designing your neural network to natural language processing. Deep learning has been used extensively in natural language processing (NLP) because
it is well suited for learning the complex underlying structure of a sentence and semantic proximity of various words.
Data cleaning, construction model (Bert, Roberta, Distilbert, XLNet, Albert, GPT, GPT2),
quality measurement training and finally visualization of your results on several dataset ( 20newsgroups, SST-2, PubMed_20k_RCT, DBPedia, Amazon Review Full, Amazon Review Polarity).


You can install it with pip :

     __pip install Manteia__

For use with GPU and cuda we recommend the use of [Anaconda](https://www.anaconda.com/open-source) :

     __conda create -n manteia_env python=3.7__

     __conda activate manteia_env__

     __conda install pytorch__

     __pip install manteia__

Example of use Classification :


	from Manteia.Classification import Classification 
	from Manteia.Model import Model 
			
	documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.','What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.']
	labels = ['funny','not funny']
			
	model = Model(model_name ='roberta')
	cl=Classification(model,documents,labels,process_classif=True)

[NoteBook](https://github.com/ym001/Manteia/blob/master/notebook/notebook_Manteia_presentation1.ipynb)


Example of use Generation :


	from Manteia.Generation import Generation 
	from Manteia.Dataset import Dataset
	from Manteia.Model import *

	
	ds=Dataset('Short_Jokes')

	model       = Model(model_name ='gpt2')
	text_loader = Create_DataLoader_generation(ds.documents_train[:10000],batch_size=32)
	model.load_type()
	model.load_tokenizer()
	model.load_class()
	model.devices()
	model.configuration(text_loader)
	
	gn=Generation(model)
	
	gn.model.fit_generation(text_loader)
	output      = model.predict_generation('What did you expect ?')
	output_text = decode_text(output,model.tokenizer)
	print(output_text)

[NoteBook](https://github.com/ym001/Manteia/blob/master/notebook/notebook_Manteia_presentation2.ipynb)

[Documentation](https://manteia.readthedocs.io/en/latest/#)
[Pypi](https://pypi.org/project/Manteia/)
[Source](https://github.com/ym001/Manteia)

This code is licensed under MIT.
