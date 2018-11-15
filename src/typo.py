# from keras import layers
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Activation
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import Embedding
# from keras.layers import Input
# from keras.layers import LSTM
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
# import keras.layers
# from keras.metrics import top_k_categorical_accuracy
# from keras.models import Model
# from keras.models import Sequential, load_model
# import keras.models
# from keras.regularizers import l2
# from keras.utils import np_utils

from CharacterTable import CharacterTable
from TypoRnnModel import TypoRnnModel
from file_process import  save_txt
from input_processor import InputProcessor
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import tensorflow as tf


# %matplotlib inline
data_path = "D:\\tensorflow\\data-2\\"

data = pd.read_csv(data_path + 'export_DATA_SET_ANALYSIS.csv', skiprows=0)
temp = data[['REGI', 'TYPO', 'VISUAL_SIMILARITY', 'SOUNDEX_DISTANCE']][(data['EDIT_DISTANCE'] == 1) & (data['IS_TYPO'] == 1) 
                                                            & ((data['VISUAL_SIMILARITY'] >= 0.8) | (data['SOUNDEX_DISTANCE'] <= 1)) 
                                                            ]
t = temp[temp.TYPO.map(lambda x: x.count('.')) == 2]
t = t[t.REGI.map(lambda x: x.count('.')) == 2]
t.reset_index(drop=True, inplace=True)
reg_list = list()
typo_list = list()
for i in range(t.shape[0]):
    reg_list.append(t['REGI'][i].split('.')[0])
    typo_list.append(t['TYPO'][i].split('.')[0])
ttt = t.drop_duplicates()
ttt.reset_index(drop=True, inplace=True)
top10 = ttt['REGI'].value_counts()[:10].index
topTypo10 = ttt['TYPO'].value_counts()[:10].index

token_size = 3
inputProcessor = InputProcessor ()
in_list , out_list = inputProcessor.processInput(typo_list, reg_list, token_size);


in_vocab = set()
out_vocab = set()
for name in in_list:
    for char in name:
        in_vocab.add(char)
for name in out_list:
    for char in name:
        out_vocab.add(char)
vocab = in_vocab.union(out_vocab)


num_encoder_tokens = len(in_vocab)
num_decoder_tokens = len(out_vocab)

table = CharacterTable(vocab)

    # print (in_list[i]+':' + out_list[i])

    
# tf.write_file(data_path+'_vocab',", ".join(vocab),None)
save_txt(data_path + '_vocab', ", ".join(vocab))
save_txt(data_path + '_in_list.txt', ", ".join(in_list))
save_txt(data_path + '_out_list.txt', ", ".join(out_list))


# Save
#np.save(data_path + 'my_file.npy', vocab) 

# Load
#vocab2 = np.load(data_path + 'my_file.npy').item()
#print(vocab2)  # displays "world"

inoutDf = pd.DataFrame({ 'A' : in_list, 'B' : out_list })
inoutDf.to_csv(data_path + 'example.csv')

is_create_model = int(input("Are you want to Create New Model? yes =1 or no = 0 "))
print ("is_create_model is: ", is_create_model)
type(is_create_model)

typoModel = TypoRnnModel (None, table, token_size);

if is_create_model == 1:
    typoModel.createModel(vocab, in_list, out_list)
    typoModel.saveModel(data_path, "final_model.hdf5")
else :
    typoModel.loadModel(data_path, "final_model.hdf5")

n = 5
num_unique = 5



for name in topTypo10:    
    unique_set = set()
    typo_dict = typoModel.predict_typo(name.split('.')[0], n)
    
    sorted_typo_dict = sorted(typo_dict, reverse=True)
    print (sorted_typo_dict)

# #    np.argsort(typo_dict)
# #
# #    keylist = mydict.keys()
# #    keylist.sort()
# #    for key in keylist:
# #        print "%s: %s" % (key, mydict[key])
    
    for prob in sorted_typo_dict[:]:
        if typo_dict[prob] != name.split('.')[0]:
            unique_set.add(typo_dict[prob])
        # if len(unique_set) == num_unique:
        #    break
    # print name.split('.')[0]
    print ('*******Generated typos***********')
    for un in unique_set:
        print(un)
    print ('\n\n')




