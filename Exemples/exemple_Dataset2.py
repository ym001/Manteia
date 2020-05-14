from Manteia.Dataset import Dataset

ds=Dataset('Yahoo! Answers')

print('Test : ')
print(ds.documents_test[:5])
print(ds.labels_test[:5])
