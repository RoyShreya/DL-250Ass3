import torch
import os
import string 
import nltk, re, pprint
import nltk
import sklearn 
import numpy as np
import scipy as sc
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
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import jsonlines
K.tensorflow_backend._get_available_gpus()

import torchnlp.datasets.snli
from nltk.corpus import stopwords
import pickle as pkl
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
import jsonlines
import numpy as np
#docList1=[]
docList=[]
doc=[]
Y=[]
a=0
b=0
c=0
X=[]
stop_words = set(stopwords.words('english'))
maxlen=0
'''
ds=nds.snli_dataset(train=True)
stop_words = set(stopwords.words('english'))
maxlen=0
import jsonlines
import joblib
for i in range(0,len(ds)):
    if ds[i]['label']=='contradiction' or ds[i]['label']=='neutral' or ds[i]['label']=='entailment' :
        words=word_tokenize(ds[i]['premise'].lower())
        words = [w for w in words if not w in stop_words]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList.append(words)
        string=" ".join(words)
        doc.append(string)
        #doc.append(words)
        words=word_tokenize(ds[i]['hypothesis'].lower())
        words = [w for w in words if not w in stop_words]
        words = [word for word in words if word.isalpha()]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList.append(words)
        string=" ".join(words)
        doc.append(string)
    
    if ds[i]['label']=='contradiction':
        Y.append(2)
    
    if ds[i]['label']=='neutral':
        Y.append(0)
    if ds[i]['label']=='entailment':
        Y.append(1)
'''
f=jsonlines.open('snli_1.0/snli_1.0/snli_1.0_train.jsonl')
for line in f.iter():

    if line['gold_label']=='contradiction' or line['gold_label']=='neutral' or line['gold_label']=='entailment' :
        words=word_tokenize(line['sentence1'].lower())
        words = [w for w in words if not w in stop_words]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList.append(words)
        string=" ".join(words)
        doc.append(string)
        #doc.append(words)
        words=word_tokenize(line['sentence2'].lower())
        words = [w for w in words if not w in stop_words]
        words = [word for word in words if word.isalpha()]
    
        if len(words)>=maxlen:
            maxlen=len(words)
        docList.append(words)
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





Y=np.array(Y)
'''
corpus = np.array(["aaa bbb ccc", "aaa bbb ddd"])
vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(corpus)
#Save vectorizer.vocabulary_
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

#Load it later
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))
'''
#Y=np_utils.to_categorical(np.array(Y))

tfidfVector=sklearn.feature_extraction.text.TfidfVectorizer(min_df=.00057,max_df=.98,decode_error="replace").fit_transform((doc))#,sparse=True)
countvec=sklearn.feature_extraction.text.CountVectorizer(min_df=.00057,max_df=.98,decode_error="replace")
countvec.fit_transform((doc))
pkl.dump(countvec.vocabulary_,open("feature.pkl","wb"))
features=tfidfVector#.todense()
i=0

while i<=(2*len(Y)-2): 
    arr=np.zeros((3))
    l=vstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    l1=hstack([csr_matrix(features[i,:]),csr_matrix(features[i+1,:])])
    #print(l.shape)
    l=l.todense()
    l1=l1.todense()
    if i==0:
        
        print(l1.shape)
    arr[0]=np.dot(l[0,:],l[1,:].T)
    arr[1]=sklearn.metrics.pairwise.euclidean_distances(l[0,:],l[1,:])
    arr[2]=sklearn.metrics.pairwise.manhattan_distances(l[0,:],l[1,:])
    #X.append(arr)
    X.append(l1)
    i=i+2
model=sklearn.linear_model.LogisticRegression()
print(np.array(X).shape)
X=np.array(X)
print(X.shape)
Y=np.array(Y)
X=np.reshape(X,(X.shape[0],X.shape[2]))  #shape of X is (549367, 1, 2398) when l1 was appended and (549367, 2398) when arr was appended

print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split( X,Y, test_size=0.30, random_state=42)
model.fit(X_train,Y_train)


# Save to file in the current working directory
joblib_file = "My_TFIDF_Modelnew1.pkl"
joblib.dump(model, joblib_file)

# Load from file
joblib_model = joblib.load(joblib_file)
print("Validation Score:")
print(joblib_model.score(X_test,Y_test))


    
