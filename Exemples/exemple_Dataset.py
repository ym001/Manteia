from Manteia.Dataset import Dataset

#ds=Dataset('20newsgroups')
#ds=Dataset('SST-2')
#ds=Dataset('SST-B')
ds=Dataset('pubmed_rct20k')

print(ds.documents_train[:5])
print(ds.labels_train[:5])
