#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Manteia.Generation import Generation 
from Manteia.Dataset import Dataset
from Manteia.Model import *

def main(args):
	
	ds=Dataset('Short_Jokes')

	model       = Model(model_name ='gpt2-medium')
	text_loader = Create_DataLoader_generation(ds.documents_train[:10000],batch_size=32)
	model.load_tokenizer()
	model.load_class()
	model.devices()
	model.configuration(text_loader)
	
	gn=Generation(model)
	
	gn.model.fit_generation(text_loader)
	output      = model.predict_generation('What did you expect ?')
	output_text = decode_text(output,model.tokenizer)
	print(output_text)
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
