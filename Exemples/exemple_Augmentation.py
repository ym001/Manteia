#modifier->parametre du notebook->GPU
#import nltk
#nltk.download('wordnet')
from Manteia.Classification import Classification 
from Manteia.Model import *
from Manteia.Dataset import Dataset
from Manteia.Preprocess import list_labels
from Manteia.Augmentation import *
from sklearn.model_selection import train_test_split,KFold

ds=Dataset('drugscom')
ds.documents_train=np.array(ds.documents_train[:100])
ds.labels_train=np.array(ds.labels_train[:100])

model = Model(model_name ='bert',early_stopping=True)
model.load_type()
model.load_tokenizer()
list_label=list_labels(ds.labels_train)
print(list_label)
model.num_labels=len(list_label)
model.load_class()
model.save('model_init')



#validation croisée
nb_pass=4
def coss_validation_idx(nb_pass,nb_docs):
  docs_idx = [idx for idx in range(nb_docs)]
  train_idx, test_idx = [], []
  for pli in range(nb_pass):
    test_pli_idx = list(np.random.choice(docs_idx,int(len(docs_idx)/nb_pass) , replace=False))
    train_pli_idx  = [idx for idx in docs_idx if idx not in test_pli_idx]
    train_idx.append(train_pli_idx)
    test_idx.append(test_pli_idx)
  return train_idx, test_idx
pli=0
acc = []
tr_idx,te_idx=coss_validation_idx(nb_pass,len(ds.documents_train))
for train_idx, test_idx in zip(tr_idx,te_idx):
  validation_idx = list(np.random.choice(train_idx,int(0.1*len(train_idx)) , replace=False))
  train_idx      = [idx for idx in train_idx if idx not in validation_idx]

  #doc_train_augmented,labels_train_augmented=eda(ds.documents_train[train_idx],ds.labels_train[train_idx])
  #doc_validation_augmented,labels_validation_augmented=eda(ds.documents_train[validation_idx],ds.labels_train[validation_idx])

  doc_train_augmented,labels_train_augmented=uda(ds.documents_train[train_idx],ds.labels_train[train_idx])
  doc_validation_augmented,labels_validation_augmented=uda(ds.documents_train[validation_idx],ds.labels_train[validation_idx])

  #doc_train_augmented,labels_train_augmented=pyramid(ds.documents_train[train_idx],ds.labels_train[train_idx])
  #doc_validation_augmented,labels_validation_augmented=pyramid(ds.documents_train[validation_idx],ds.labels_train[validation_idx])
	
  train_ids,train_masks           = encode_text(doc_train_augmented,model.tokenizer,model.MAX_SEQ_LEN)
  train_labels                    = encode_label(labels_train_augmented,list_label)

  validation_ids,validation_masks = encode_text(doc_validation_augmented,model.tokenizer,model.MAX_SEQ_LEN)
  validation_labels               = encode_label(labels_validation_augmented,list_label)

  test_ids,test_masks             = encode_text(ds.documents_train[test_idx],model.tokenizer,model.MAX_SEQ_LEN)
  test_labels                     = encode_label(ds.labels_train[test_idx],list_label)

  dt_train       = Create_DataLoader(train_ids,train_masks,train_labels)
  dt_validation  = Create_DataLoader(validation_ids,validation_masks,validation_labels)
  dt_test        = Create_DataLoader(test_ids,test_masks)
  
  model.load('model_init')

  model.configuration(dt_train)
  model.fit(dt_train,dt_validation)

  predictions=model.predict(dt_test,p_type='class')
  acc.append(accuracy(test_labels,predictions))
  print('Passe cross validation : {} , Accuracy : {}'.format(pli,acc[-1]))
  pli+=1

print('')
print('Cross validation accuracy : {} : {}'.format(np.mean(np.array(acc)),acc))
