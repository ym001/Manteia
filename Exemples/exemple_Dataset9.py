from Manteia.Dataset import Dataset

ds=Dataset('trec')

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])


