from Manteia.Classification import Classification 
from Manteia.Model import Model 
			
documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.','What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
labels = ['funny','not funny']
			
model = Model(model_name ='bert')
cl=Classification(model,documents,labels,process_classif=True)
