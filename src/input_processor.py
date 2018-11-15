import numpy as np
'''
Created on Nov 11, 2018

@author: User
'''

class InputProcessor(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
    def levenshtein_distance(self, string1, string2):
        n1 = len(string1)
        n2 = len(string2)
        return self._levenshtein_distance_matrix(string1, string2)[n1, n2]
    
    def damerau_levenshtein_distance(self, string1, string2):
        n1 = len(string1)
        n2 = len(string2)
        return self._levenshtein_distance_matrix(self, string1, string2, True)[n1, n2]
    
    
    def _levenshtein_distance_matrix(self, string1, string2, is_damerau=False):
        n1 = len(string1)
        n2 = len(string2)
        d = np.zeros((n1 + 1, n2 + 1), dtype=int)
        for i in range(n1 + 1):
            d[i, 0] = i
        for j in range(n2 + 1):
            d[0, j] = j
        for i in range(n1):
            for j in range(n2):
                if string1[i] == string2[j]:
                    cost = 0
                else:
                    cost = 1
                d[i + 1, j + 1] = min(d[i, j + 1] + 1,  # insert
                                  d[i + 1, j] + 1,  # delete
                                  d[i, j] + cost)  # replace
                if is_damerau:
                    if i > 0 and j > 0 and string1[i] == string2[j - 1] and string1[i - 1] == string2[j]:
                        d[i + 1, j + 1] = min(d[i + 1, j + 1], d[i - 1, j - 1] + cost)  # transpose
        return d

    def get_ops(self, string1, string2, is_damerau=False):
        dist_matrix = self._levenshtein_distance_matrix(string1, string2, is_damerau)
        i, j = dist_matrix.shape
        i -= 1
        j -= 1
        ops = list()
        while i != -1 and j != -1:
            if is_damerau:
                if i > 1 and j > 1 and string1[i - 1] == string2[j - 2] and string1[i - 2] == string2[j - 1]:
                    if dist_matrix[i - 2, j - 2] < dist_matrix[i, j]:
                        ops.insert(0, ('transpose', i - 1, i - 2))
                        i -= 2
                        j -= 2
                        continue
            index = np.argmin([dist_matrix[i - 1, j - 1], dist_matrix[i, j - 1], dist_matrix[i - 1, j]])
            if index == 0:
                if dist_matrix[i, j] > dist_matrix[i - 1, j - 1]:
                    ops.insert(0, ('replace', i - 1, j - 1))
                i -= 1
                j -= 1
            elif index == 1:
                ops.insert(0, ('insert', i - 1, j - 1))
                j -= 1
            elif index == 2:
                ops.insert(0, ('delete', i - 1, i - 1))
                i -= 1
        return ops    
        
    def processInput(self, reg_list, typo_list, token_size):
        in_list = list()
        out_list = list()
        # i=26
        for i in range(len(reg_list)):
            ops = self.get_ops(reg_list[i], typo_list[i])
            if len(ops) == 0:
                continue
            if ops[0][0] == 'replace':
                for start_idx in range(ops[0][1] - token_size + 1, ops[0][1] + 1):
                    if start_idx < 0 :
                        continue
                    if start_idx > (len(reg_list[i]) - token_size):
                        break
                    in_list.append('#' + reg_list[i][start_idx:start_idx + token_size] + '$')
                    out_list.append('#' + typo_list[i][start_idx:start_idx + token_size] + '$')
                
            elif ops[0][0] == 'insert':
                for start_idx in range(ops[0][1] - token_size + 1, ops[0][1] + 1):
                    if start_idx < 0:
                        continue
                    if start_idx > (len(reg_list[i]) - token_size):
                        continue
                    in_list.append('#' + reg_list[i][start_idx:start_idx + token_size] + '$')
                    out_list.append('#' + typo_list[i][start_idx:start_idx + token_size + 1])
                for start_idx in range(ops[0][1] + 1, ops[0][1] + 2):
                    if start_idx > (len(reg_list[i]) - token_size):
                        continue
                    in_list.append('#' + reg_list[i][start_idx:start_idx + token_size] + '$')
                    out_list.append(typo_list[i][start_idx:start_idx + token_size + 1] + '$')
        
        
            elif ops[0][0] == 'delete':
                tt = reg_list[i][:ops[0][1]] + '*' + typo_list[i][ops[0][1]:]
                for start_idx in range(ops[0][1] - token_size + 1, ops[0][1] + 1):
                    if start_idx < 0 :
                        continue
                    if start_idx > (len(reg_list[i]) - token_size):
                        break
                    in_list.append('#' + reg_list[i][start_idx:start_idx + token_size] + '$')
                    out_list.append('#' + tt[start_idx:start_idx + token_size] + '$')
        
                
            else:
                print ('*******************')
                

        
        return in_list, out_list;
