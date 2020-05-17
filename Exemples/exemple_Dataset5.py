from Manteia.Dataset import Dataset

ds=Dataset('20newsgroups')

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])
