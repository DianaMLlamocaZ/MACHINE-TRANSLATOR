import torch
import numpy as np
import pandas as pd
import unicodedata
import re

#Device
device="cuda" if torch.cuda.is_available() else "cpu"


#Read Dataset
def read_dataset(path_dataset):
  dataframe=pd.read_csv(path_dataset)
  return dataframe

datapath="../dataset/data.csv"

dataset=read_dataset(datapath)


#Normalización data:
#-Descomposición de caracteres unicode
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

#-Minúscula, trim y remover caracteres que NO son letras
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.¡!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.¡!?]+", r" ", s)
    return s.strip()


#Idioma tokens
class Idioma():
  def __init__(self):
    self.word2index={"pad":0,"sos":1,"eos":2,"unk":3}
    self.index2word={0:"pad",1:"sos",2:"eos",3:"unk"}
    self.word_count={}

  def tokenizar(self,oracion):

    #normalización
    sentence_norm=normalizeString(oracion).split(" ")

    for word in sentence_norm:

      #Si la NO palabra está en el diccionario, actualizo los diccionarios correspondientes
      if word not in self.word_count:
        self.word_count[word]=1
        self.word2index[word]=int(len(self.word2index))
        self.index2word[self.word2index[word]]=word


      #Si está, entonces se coloca '1'
      else:
        self.word_count[word]+=1


#Create idioma class
def create_idiom_class(dataset):
  #Clases que representan a cada idioma
  idioma1=Idioma() #English
  idioma2=Idioma() #Spanish

  #Recorro el dataframe para tokenizar las oraciones
  for i in range(dataset.shape[0]):
    eng,sp=dataset.iloc[i]["english"],dataset.iloc[i]["spanish"]
    norm_eng,norm_sp=normalizeString(eng),normalizeString(sp)

    if len(norm_eng.split(" "))<=20 and len(norm_sp.split(" "))<=20: #<=10
      idioma1.tokenizar(norm_eng) #Tokenizando la oración en inglés
      idioma2.tokenizar(norm_sp) #Tokenizar la oración en español


  return idioma1,idioma2


#Tokenizer
def tokenize_sentence(idioma,oracion_norm):
  tokens_emb=[1] #Token SOS

  for word in oracion_norm.split(" "):
    tokens_emb.append(idioma.word2index[word])

  tokens_emb.append(2) #Token EOS

  return tokens_emb


#Diccionarios para mapear tokens a palabras (de ambos idiomas)
id1,id2=create_idiom_class(dataset)
