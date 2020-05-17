from Manteia.Dataset import Dataset

ds=Dataset('SST-2')

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])


