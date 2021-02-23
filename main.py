from  Trainer import Model_trainer
from Organizer import cleaner
from Utiltites import utlities
from generator import generator
lenght = 100 + 1
filename = 'Furry_Text.txt'
New_filename = './NewText.txt'
model = 'model.h5'

ult = utlities(filename,New_filename)

# #Orginize Text for model traing
cl = cleaner(len=lenght,file=ult.load_text())
ult.save(cl.organize())




# #Training Model
Trainer = Model_trainer(New_filename,model,1)
Trainer.train_words()
# #justt what to figure out how predction worked
Gen = generator('model.h5','tokenizer_char.pkl',99,'all talk and no action',60)



