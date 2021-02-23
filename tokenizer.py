from keras.preprocessing.text import Tokenizer
from keras.utils import  to_categorical
import numpy as np
from pickle import dump
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text

def load(file):
    file = open(file,'r')
    text = file.read()
    file.close()
    return text

class tokenmaker:
    text = None

    def __init__(self,file):
        self.text = load(file)
        

    def tokenize(self):
        print('=======================initalizing tokenizer==========================')
        lines = self.text.split('\n')
        tokenizer= Tokenizer()
        tokenizer.fit_on_texts(lines)
        sequences = tokenizer.texts_to_sequences(lines)
        #Beccause indexing of arrays is zero-offest,the index of word at the end of the vocaubulary will be (len(vocab)) that means the array must be len(vocab) + 1 
        vocab_size = len(tokenizer.word_index) + 1
        sequences = np.array(sequences)
        X,Y = sequences[:,:-1],sequences[:,-1]
        Y = to_categorical(Y,num_classes=vocab_size)
        seq_length = X.shape[1]
        print('=======================saving tokenizer==========================')
        dump(tokenizer, open('tokenizer.pkl', 'wb'))
        print('=======================done tokenizering==========================')
        return X,Y,seq_length,vocab_size

    def tokenize_chars(self):
        lines = self.text.split('\n')
        
        alphabet = 'abcdefghijkmlnopqrstuvwkyz1234576890`?><:/*-+[[;./,'
        print('=======================initalizing tokenizer(chars)==========================')
        tokenizer= Tokenizer(char_level=True,oov_token='UNK')
        tokenizer.fit_on_texts(lines)
        print(tokenizer.word_index)
        sequences = tokenizer.texts_to_sequences(lines)
        data = pad_sequences(sequences,maxlen=100,padding='post')
        print(sequences[0])
        char_size = len(tokenizer.word_index) + 1
        print(char_size)
        sequences = np.array(data)
        X = sequences[:,:-1]
        Y= sequences[-1]
        Y = to_categorical(Y,num_classes=char_size)
        seq_length = X.shape[1]
        print('=======================saving tokenizer(chars)==========================')
        dump(tokenizer, open('tokenizer_char.pkl', 'wb'))
        print('=======================done tokenizering(chars)==========================')
        return X,Y,seq_length,char_size

    def tokenize_csv(self):
        x1,x2 = ult.import_csv('trump_insult_tweets_2014_to_2021.csv')
        lines1 = x1
        print(type(x1))
        lines2 = x2
        lines = lines1 +lines2
        tokenizer= Tokenizer()
        tokenizer.fit_on_texts(lines)
        sequences1 = tokenizer.texts_to_sequences(lines1)
        sequences2 = tokenizer.texts_to_sequences(lines2)
        data1 = pad_sequences(sequences1,maxlen=100,padding='post')
        data2 = pad_sequences(sequences2,maxlen=100,padding='post')
        print(sequences1[0])
        char_size = len(tokenizer.word_index) + 1
        print(char_size)
        sequences1 = np.array(data1)
        sequences2 = np.array(data2)
        X1 = sequences1[:,:-1]
        X2 = sequences2[:,:-1]
        Y= sequences1[:,-1]
        Y = to_categorical(Y,num_classes=char_size)
        seq_length1 = X1.shape[1]
        seq_length2 = X2.shape[1]
        print('=======================saving tokenizer(chars)==========================')
        dump(tokenizer, open('tokenizer_char.pkl', 'wb'))
        print('=======================done tokenizering(chars)==========================')
        return X1,X2,Y,seq_length1,seq_length2,char_size

