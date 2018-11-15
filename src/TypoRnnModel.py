import numpy as np
import keras.models
import keras.layers
from keras import layers
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import LSTM

class TypoRnnModel(object):

    def __init__(self, model, table , token_size):
        self.model = model;
        self.table = table
        self.token_size = token_size

    
    def get_pred(self, pred, depth=2):
        sorted_idx = np.argsort(-pred)
        d = dict()
        for i0 in range(depth):
            idx0 = sorted_idx[0, 0, i0]
            p0 = pred[0, 0, idx0]
            c0 = self.table.indices_char[idx0]
    
            for i1 in range(depth):
                idx1 = sorted_idx[0, 1, i1]
                p1 = p0 * pred[0, 1, idx1]
                c1 = c0 + self.table.indices_char[idx1]
                # print c1
    
                for i2 in range(depth):
                    idx2 = sorted_idx[0, 2, i2]
                    p2 = p1 * pred[0, 2, idx2]
                    c2 = c1 + self.table.indices_char[idx2]
                    # print c2,p2
    
                    for i3 in range(depth):
                        idx3 = sorted_idx[0, 3, i3]
                        p3 = p2 * pred[0, 3, idx3]
                        c3 = c2 + self.table.indices_char[idx3]
                        # print c3,p3
                        # d[p3]=c3
    
                        for i4 in range(depth):
                            idx4 = sorted_idx[0, 4, i4]
                            p4 = p3 * pred[0, 4, idx4]
                            c4 = c3 + self.table.indices_char[idx4]
                            d[p4] = c4
                            if p4 < 0.001:
                                break
                                
        # d[0] = table.indices_char[sorted_idx[0,0,0]] + table.indices_char[sorted_idx[0,1,0]] + table.indices_char[sorted_idx[0,2,0]] + table.indices_char[sorted_idx[0,3,0]] + table.indices_char[sorted_idx[0,4,0]];
        return d  
    
    def predict_typo(self, domain_name, n):
        typo_dict = dict()
        t_dom_count = 0;
        for i in range(len(domain_name) - self.token_size + 1):
            substr = domain_name[i:i + self.token_size]
            substr = '#' + substr + '$'
            # print (substr)
            pred = self.model.predict(self.table.encode(substr, self.token_size + 2).reshape((1, self.token_size + 2, len(self.table.chars))))
            # print (pred)
            d = self.get_pred(pred, n)
            ss = sorted(d, reverse=True)
            
            for val in ss[:n]:
                pred_sub_str = d[val]
                # print (pred_sub_str)
                if pred_sub_str[-1] == '$':
                    pred_sub_str = pred_sub_str[:-1]
                if pred_sub_str[0] == '#':
                    pred_sub_str = pred_sub_str[1:]
                
                pred_str = pred_sub_str.replace("*", "");
                pred_typo = domain_name[:i] + pred_str + domain_name[i + self.token_size:]
                typo_dict[val] = pred_typo
                t_dom_count = t_dom_count + 1;
            # print ('predict>' , typo_dict)
        return typo_dict;
    def encodeInputOutputList(self, in_list, out_list, vocab):
        
        max_encoder_seq_length = max([len(name) for name in in_list])
        max_decoder_seq_length = max([len(name) for name in out_list])
        encoder_input_data = np.zeros((len(in_list), max_encoder_seq_length, len(vocab)), dtype='float32')
        decoder_input_data = np.zeros((len(out_list), max_decoder_seq_length, len(vocab)), dtype='float32')
        for i in range(len(in_list)):
            encoder_input_data[i] = self.table.encode(in_list[i], self.token_size + 2)
            decoder_input_data[i] = self.table.encode(out_list[i], self.token_size + 2)
        return encoder_input_data , decoder_input_data
    def createModel(self, vocab , in_list, out_list):
        print('Build model...')
        encoder_input_data , decoder_input_data = self.encodeInputOutputList(in_list, out_list, vocab)
        RNN = layers.LSTM
        HIDDEN_SIZE = 40
        BATCH_SIZE = 16
        LAYERS = 2    
        self.model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
        # Note: In a situation where your input sequences have a variable length,
        # use input_shape=(None, num_feature).
        self.model.add(RNN(HIDDEN_SIZE, input_shape=(self.token_size + 2, len(vocab))))
        # As the decoder RNN's input, repeatedly provide with the last hidden state of
        # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
        # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
        self.model.add(layers.RepeatVector(self.token_size + 2))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(LAYERS):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            self.model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    
        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        self.model.add(layers.TimeDistributed(layers.Dense(len(vocab))))
        self.model.add(layers.Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.summary()
    
        print (len(vocab))
        history = self.model.fit(encoder_input_data, decoder_input_data,
                      batch_size=BATCH_SIZE,
                      epochs=100,
                      validation_split=0.2,
                        # callbacks = callbacks_list,
                      verbose=True)
        return self.model;
    
    def saveModel(self, filePath, fileName):
        self.model.save(filePath + fileName)
        return self.model;
        
    def loadModel(self, filePath, fileName):
        print('load model...')
        self.model = load_model(filePath + fileName)
        return self.model;
