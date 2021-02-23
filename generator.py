from pickle import load
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class  generator:

    def __init__(self,model,tokenizer,seq_length,seed_text,n_words):
        self.model =load_model(model)
        self.tokenizer = tokenizer = load(open(tokenizer, 'rb'))
        self.seq_length= seq_length
        self.seed_text = seed_text
        self.n_words= n_words


    def generate_seq(self):
        result = list()
        in_text = self.seed_text
        # generate a fixed number of words
        for _ in range(self.n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=self.seq_length, truncating='pre')
            # predict probabilities for each word
            yhat = self.model.predict([encoded,encoded])
            y_classes = yhat.argmax(-1)
            #.predict_classes(encoded, verbose=0)
            # map predicted word index to word
            out_word = ''
            for word, index in self.tokenizer.word_index.items():
                if index == y_classes:
                    out_word = word
                    break
            # append to input
            in_text += ' ' + out_word
            result.append(out_word)
        return ' '.join(result)