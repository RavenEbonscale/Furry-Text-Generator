import  pandas as pd
class utlities:
    file = None
    newfile = None

    def __init__(self,file,newfile):
        self.file = file
        self.newfile = newfile
    
    def load_text(self):
        print('++++opening file+++++')
        file = open(self.file,'r',encoding='utf-8')
        text= file.read()
        file.close()
        print(text[:200])
        return text
    
    def save(self,sequence):
        print('++++saving file+++++')
        lines = sequence
        data = '\n'.join(lines)
        file= open(self.newfile, 'w')
        file.write(data)
        file.close()
    
    def import_csv(self,csvfile):
        col_list = ['insult','tweet']
        Training_data = pd.read_csv(csvfile,header=0,usecols=col_list ,skipinitialspace=True,sep=',')
        Training_data.head()
        x1 = Training_data['insult'].tolist()
        x2 = Training_data['tweet'].tolist()
        return x1,x2