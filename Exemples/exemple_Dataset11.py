from Manteia.Dataset import Dataset

ds=Dataset('SST-5',dev=True)

print('Dev : ')
print(ds.documents_dev[:5])
print(ds.labels_dev[:5])


