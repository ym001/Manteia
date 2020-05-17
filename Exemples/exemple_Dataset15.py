from Manteia.Dataset import Dataset

ds=Dataset('Tweeter Airline Sentiment')

print('Train : ')
print(ds.documents_train[:5])
print(ds.labels_train[:5])
