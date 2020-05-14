from Manteia.Dataset import Dataset

ds=Dataset('pubmed_rct20k')

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])

