from Manteia.Classification import Classification 
from Manteia.Preprocess import list_labels 
			
documents = ['What should you do before criticizing Pac-Man? WAKA WAKA WAKA mile in his shoe.','What did Arnold Schwarzenegger say at the abortion clinic? Hasta last vista, baby.',]
			
labels = ['funny','not funny']
			
cl=Classification(documents_train = documents,labels_train = labels)
cl.list_labels     = list_labels(labels)
cl.load_model()
dt_train ,dt_validation=cl.process_text()
cl.model.configuration(dt_train)
cl.model.fit(dt_train,dt_validation)
