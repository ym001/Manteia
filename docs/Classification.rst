Classification
==============

.. automodule:: Manteia.Classification
    :members:
    
A complete example
==================


    from Manteia.Classification import Classification 
    >>>from Manteia.Preprocess import Preprocess

.. code-block:: python
   :linenos:


	from Manteia.Dataset import Dataset
	from Manteia.Classification import Classification 
	from Manteia.Preprocess import Preprocess
	
	def main(args):
	
		ds             = Dataset('20newsgroups')
		documents      = ds.get_documents()
		labels         = ds.get_labels()
		pp             = Preprocess(documents=documents,labels=labels,nb_sample=500)
		documents      = pp.documents
		labels         = pp.labels
		cl             = Classification(documents=documents,labels=labels)
		cl.list_labels = pp.list_labels

		cl.load_model()
		dt_train ,dt_validation = cl.process_text()
		cl.model.configuration(dt_train)
		cl.model.fit(dt_train,dt_validation)
			
		print(cl.predict(documents[:5]))

		return 0

	if __name__ == '__main__':
		import sys
		sys.exit(main(sys.argv))
    
   code
