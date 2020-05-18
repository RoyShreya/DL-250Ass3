from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import os
import string 
import nltk, re, pprint
import nltk
import sklearn 
import numpy as np
import scipy as sc
import pickle
import pickle as pkl
from nltk.corpus import stopwords
import torchnlp.datasets as nds
from gensim.models.word2vec import Word2Vec
import torchnlp.datasets.snli
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from nltk import word_tokenize
from keras.layers import Flatten,LSTM,Dense,Concatenate,Input,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.optimizers import *
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
import torchnlp.datasets as nds
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset,DataLoader
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer

import jsonlines
import numpy as np
docList1=[]
docList2=[]
doc=[]
Y=[]
#ds=nds.snli_dataset(test=True)
stop_words = set(stopwords.words('english'))
maxlen=49
import jsonlines

f=jsonlines.open('snli_1.0/snli_1.0/snli_1.0_test.jsonl')
a=0
b=0
c=0

for line in f.iter():

    if line['gold_label']=='contradiction' or line['gold_label']=='neutral' or line['gold_label']=='entailment' :
        words=word_tokenize(line['sentence1'].lower())
        words = [w for w in words if not w in stop_words]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList1.append(words)
        string=" ".join(words)
        doc.append(string)
        #doc.append(words)
        words=word_tokenize(line['sentence2'].lower())
        words = [w for w in words if not w in stop_words]
        words = [word for word in words if word.isalpha()]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList2.append(words)
        string=" ".join(words)
        doc.append(string)
    
    if line['gold_label']=='contradiction':
        Y.append(2)
        a=a+1
    if line['gold_label']=='neutral':
        Y.append(0)
        b=b+1
    if line['gold_label']=='entailment':
        Y.append(1)
        c=c+1
Y_test=Y#TFIDF model will use Y_test
Y=np_utils.to_categorical(np.array(Y))

#Y=np.array(Y)

#t = Tokenizer()
t=pickle.load(open('wordtovec.pkl','rb'))
encoded_docs = t.texts_to_sequences(doc)

padded_docs = pad_sequences(encoded_docs, maxlen=49, padding='post')

vocab_size=len(np.unique(np.array(padded_docs)).tolist())

leftX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))
rightX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))
#1int(padded_docs.shape[0]/2)
Y=np.array(Y)

for i in range(0,padded_docs.shape[0]-1,2):
    
    leftX[int(i/2),:]=padded_docs[i,:]
    rightX[int(i/2),:]=padded_docs[i+1,:]
    #else:
        

model=load_model('best_model.hdf5')
#model=load_model('MyBidirLSTM2Layer100Dim.h5')
X=np.concatenate((leftX,rightX),axis=1)
o=model.evaluate(X,Y)[1]
print("Test Accuracy by my deep model")
print(o)

transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
features = transformer.fit_transform(loaded_vec.fit_transform(np.array(doc)))
i=0
X_test=[]
while i<=(2*len(Y)-2): 
    arr=np.zeros((3))
    l=vstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    l1=hstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    #print(l.shape)
    l=l.todense()
    l1=l1.todense()
    
    arr[0]=np.dot(l[0,:],l[1,:].T)
    arr[1]=sklearn.metrics.pairwise.euclidean_distances(l[0,:],l[1,:])
    arr[2]=sklearn.metrics.pairwise.manhattan_distances(l[0,:],l[1,:])
    #X.append(arr)
    X_test.append(l1)
    i=i+2
joblib_model = joblib.load('My_TFIDF_Modelnew1.pkl')
X_test=np.array(X_test)
#print(X_test.shape)    # shape is (9824, 1, 2398) if l1 was added else (9824, 2398) if arr was added


X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[2]))  #
print("Test Score by TFIDF feature engineering model:")
print(joblib_model.score(X_test,Y_test))



###### generate output for all test samples - this time loading all test samples:



f=jsonlines.open('snli_1.0/snli_1.0/snli_1.0_test.jsonl')
a=0
b=0
c=0

for line in f.iter():

    if 1==1 :
        words=word_tokenize(line['sentence1'].lower())
        words = [w for w in words if not w in stop_words]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList1.append(words)
        string=" ".join(words)
        doc.append(string)
        #doc.append(words)
        words=word_tokenize(line['sentence2'].lower())
        words = [w for w in words if not w in stop_words]
        words = [word for word in words if word.isalpha()]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList2.append(words)
        string=" ".join(words)
        doc.append(string)
    
encoded_docs = t.texts_to_sequences(doc)

padded_docs = pad_sequences(encoded_docs, maxlen=49, padding='post')

vocab_size=len(np.unique(np.array(padded_docs)).tolist())

leftX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))
rightX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))


for i in range(0,padded_docs.shape[0]-1,2):
    
    leftX[int(i/2),:]=padded_docs[i,:]
    rightX[int(i/2),:]=padded_docs[i+1,:]  
X=np.concatenate((leftX,rightX),axis=1)  
pred=model.predict(X)

print(pred.shape)
out=np.argmax(pred,axis=1)

output=[]
for i in range (0,pred.shape[0]):
   if out[i]==2:
        output.append('contradiction')
   else:
        if out[i]==0:
            output.append('neutral')
        else:
            output.append('entailment')
#print(output)
np.savetxt('deep_model.txt', output, fmt='%s', newline='\n')
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
features = transformer.fit_transform(loaded_vec.fit_transform(np.array(doc)))
i=0
X_test=[]
while i<=(2*len(Y)-2): 
    arr=np.zeros((3))
    l=vstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    l1=hstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    #print(l.shape)
    l=l.todense()
    l1=l1.todense()
    
    arr[0]=np.dot(l[0,:],l[1,:].T)
    arr[1]=sklearn.metrics.pairwise.euclidean_distances(l[0,:],l[1,:])
    arr[2]=sklearn.metrics.pairwise.manhattan_distances(l[0,:],l[1,:])
    #X.append(arr)
    X_test.append(l1)
    i=i+2
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[2]))  #
pred=joblib_model.predict(X_test)

out=pred

output=[]
for i in range (0,out.shape[0]):
   if out[i]==2:
        output.append('contradiction')
   else:
        if out[i]==0:
            output.append('neutral')
        else:
            output.append('entailment')
#print(output)
np.savetxt('tfidf.txt', output, fmt='%s', newline='\n')



