from Manteia.Dataset import Dataset

ds=Dataset('20newsgroups')
documents=ds.get_documents()
labels=ds.get_labels()

print(documents[:5])
print(labels[:5])
