from sys import path
from keras.utils import plot_model
from keras.models import Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import  Dense,LSTM,Embedding,Dropout,Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.saving.save import load_model
from keras.layers.merge import concatenate
from tokenizer import tokenmaker
import  os.path
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)






class Model_trainer:

    def __init__(self,file,model_path,epochs):
        T = tokenmaker(file)
        self.epochs = epochs
        self.X_w,self.Y_w,self.seq_length_w,self.vocab_size_w= T.tokenize()
        #self.X_c,self.Y_c,self.seq_length_c,self.vocab_size_c = T.tokenize_chars('Comments.txt')
        #self.input1,self.input2,self.y,self.seq_length_1,self.seq_length_2,self.vocab_size_w = T.tokenize_csv()
        self.checkpoint = ModelCheckpoint('model.h5',monitor='accuracy',verbose=1,save_best_only=True,mode='max')
        self.callback_list = [self.checkpoint]
        self.model =model_path
        self.Adam=Adam(lr=.1)
        self.Adam=tf.train.experimental.enable_mixed_precision_graph_rewrite(self.Adam)


    def Create_model_words(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size_w,20,input_length=self.seq_length_w))
        model.add(LSTM(128,return_sequences=True))
        #model.add(Dropout(.1))
        model.add(LSTM(128))
        # model.add(Dropout(.5))
        # model.add(Dense(100,activation='relu'))
        # model.add(Dropout(.5))
        # model.add(LSTM(100))
        # model.add(Dropout(.5))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(self.vocab_size_w,activation='softmax'))
        print(model.summary())
        return model

    def Create_model_chars(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size_c,50,input_length=self.seq_length_c))
        model.add(LSTM(100,return_sequences=True))
        model.add(Dropout(.5))
        model.add(LSTM(100,return_sequences=True))
        model.add(Dropout(.5))
        model.add(Dense(100,activation='relu'))
        model.add(Dropout(.5))
        model.add(LSTM(100))
        model.add(Dropout(.5))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(self.vocab_size_c,activation='softmax'))
        print(model.summary())
        return model
    
    def Create_Functional(self):
        #first input layer thing
        input1 = Input(shape=(self.seq_length_1,))
        embeding1 = Embedding(self.vocab_size_w,100,input_length=self.seq_length_1)(input1)
        ltsm1= LSTM(128)(embeding1)
        hidden1= Dense(100,activation='relu')(ltsm1)
        #second input
        input2 =Input(shape=(self.seq_length_2,))
        embeding21 = Embedding(self.vocab_size_w,100,input_length=self.seq_length_2)(input2)
        ltsm21= LSTM(128)(embeding21)
        hidden2= Dense(10,activation='relu')(input2)
        #merge inputs
        merge= concatenate([hidden1,hidden2])
        output = Dense(self.vocab_size_w,activation='softmax')(merge)
        model = Model(inputs=[input1, input2], outputs=output)
        print(model.summary())

        return model
        #plot_model(model, to_file='multiple_inputs.png') this is broken

    def train_words(self):
        if os.path.exists(self.model):
            model = load_model(self.model)
        else:
            model= self.Create_model_words()
        model.summary()
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(self.X_w,self.Y_w,batch_size=64,epochs=self.epochs,callbacks=self.callback_list)
        model.save('model.h5')

    def train_chars(self):
        model = self.Create_model_chars()
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(self.X_c,self.Y_c,batch_size=64,epochs=self.epochs,callbacks=self.callback_list)
        model.save('model_chars.h5')

    def train_users(self):
        model = self.Create_Functional()
        model.save('model_t.h5')
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit([self.input1,self.input2],self.y,batch_size=64,epochs=self.epochs,callbacks=self.callback_list,)
        model.save('model_t.h5')
        