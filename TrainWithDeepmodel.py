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
import pickle
from gensim.models.word2vec import Word2Vec
import torchnlp.datasets.snli
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from nltk import word_tokenize
from keras.layers import Flatten,LSTM,Dense,Concatenate,Input,BatchNormalization,Dropout,Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import jsonlines
K.tensorflow_backend._get_available_gpus()
import numpy as np
docList1=[]
docList2=[]
doc=[]
Y=[]
#ds=nds.snli_dataset(train=True)
stop_words = set(stopwords.words('english'))
maxlen=0
checkpoint = ModelCheckpoint("best_modelWithjson.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=1)
f=jsonlines.open('snli_1.0/snli_1.0/snli_1.0_train.jsonl')
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
    
   
print(a)
print(b)
print(c)

Y=np_utils.to_categorical(np.array(Y))
print(doc[0])
print(doc[1])
#Y=np.array(Y)
output_dim=512
num_classes=3
t = Tokenizer()
t.fit_on_texts(doc)

with open('wordtovec.pkl', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
encoded_docs = t.texts_to_sequences(doc)

padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')
print(maxlen)
print(padded_docs.shape)
vocab_size=len(np.unique(np.array(padded_docs)).tolist())

leftX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))
rightX=np.zeros((int(padded_docs.shape[0]/2), padded_docs.shape[1]))#,padded_docs.shape[2]))
#1int(padded_docs.shape[0]/2)
Y=np.array(Y)

print(Y[1])
for i in range(0,padded_docs.shape[0]-1,2):
    print(i)
    leftX[int(i/2),:]=padded_docs[i,:]
    rightX[int(i/2),:]=padded_docs[i+1,:]
    #else:
        
print(leftX[0])
print(rightX[0])
print(leftX[1])
print(rightX[1])

X=np.concatenate((leftX,rightX),axis=1)
print(X.shape)
print(X[0])
print(X[1])
print(Y[0])
embeddings_index = dict()
f = open('glove.6B.200d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 200))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
#model1.add(Input(batch_shape=(81,maxlen),dtype='int32'))
#lstm_in2=Input(batch_shape=(81,time_seq,vec_len))
e = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=2*maxlen, trainable=False)
model.add(e)

model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))#,stateful=True))
#model.add(Dropout(.2))
model.add(Bidirectional(LSTM(output_dim, return_sequences=False)))
#model.add(Dropout(.3))
model.add(Dense(output_dim=512, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(output_dim=512, activation='relu'))
model.add(Dense(output_dim=num_classes, activation='softmax'))

num_epochs=20
opt=Adam(lr=.0001)#(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#(lr=.005)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
print('Going to train')
#history=model.fit(inputs=[leftX_train, rightX_train], outputs=Y,shuffle=True,batch_size=54, validation_split=.33)
#history=model.fit(x=[leftX, rightX],y=Y,shuffle=True, validation_split=.25,epochs=num_epochs,batch_size=81) #batch_size=81,
history=model.fit(x=X,y=Y,shuffle=True, validation_split=.25,epochs=num_epochs,batch_size=81,callbacks=[checkpoint]) #batch_size=81,
model.save('MyBidirLSTM2Layer200DimFinalWithjson.h5')
val_loss   = history.history['val_loss']

#print(val_loss)
np.save('val_loss3FWithjson.npy',val_loss)

train_loss=history.history['loss']
np.save('train_loss3FWithjson.npy',train_loss)
val_acc=history.history['val_acc']
np.save('val_acc3FWithjson.npy',val_acc )
train_acc=history.history['acc']

np.save('train_acc3FWithjson.npy',train_acc )

