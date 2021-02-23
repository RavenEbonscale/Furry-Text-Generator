import string
import pandas as pd


class cleaner:
    file =None
    length = 0
    #length - 50+1

    def __init__(self,len,file):
        self.length = len
        self.file = file

    def clean_text(self,doc):
        doc = doc.replace('--','')
        tokens = doc.split()
        table= str.maketrans('','',string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens= [word.lower() for word in tokens]
        print(tokens[:200])
        print(f'Total Tokens: {len(tokens)}')
        print(f'Unique Tokens: {len(set(tokens))}')
        return tokens


    def organize(self):
        token = self.clean_text((self.file))
        sequences = list()
        for i in range(self.length,len(token)):
            seq = token[i-self.length:i]
            line = ' '.join(seq)
            sequences.append(line)
        print(f'Total Sequences:{len(sequences)}')
        return sequences

    def organize_chars(self):
        text = self.file.replace("\n", " ")  # We remove newlines chars for nicer display
        print("Corpus length:", len(text))
        chars = sorted(list(set(text)))
        print("Total chars:", len(chars))
        return chars
