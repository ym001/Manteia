from Manteia.Dataset import Dataset

ds=Dataset('Yelp Review Polarity',test=True,desc=True)

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])
print(ds.documents_test[:5])
print(ds.labels_test[:5])
print(ds.description)
