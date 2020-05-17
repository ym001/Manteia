from Manteia.Dataset import Dataset

ds=Dataset('DBPedia',test=True,desc=True,classe=True)

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])

print('Test : ')
print(ds.documents_test[:5])
print(ds.labels_test[:5])

print('Description :')
print(ds.description)

print('List labels :')
print(ds.list_labels)
