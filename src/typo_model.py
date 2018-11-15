import pandas as pd
import numpy as np

import datetime

def preporcess(str):
    domain= str.lower().replace("www.", "")
    dompart = domain.split('.')
    
    if len(dompart) == 3:
        return domain
    else:
        return ''
    

data = []  # create an empty list to collect the data
# open the file and read through it line by line
with open("F:\\domain-log\\2017-11\\112\\queries.log-20171101", 'r') as file_object:
    line = file_object.readline()
    data = {}
    while line:
        # at each line check for a match with a regex
        step = line.split(' ');

        
        step1=step[5].split('#')
        
        
        #print (step1[0],step[0], step[1], step[8])
        date_time_str = step[0] + ' ' + step[1]  
        date_time_obj = datetime.datetime.strptime(date_time_str, '%d-%b-%Y %H:%M:%S.%f')
        
        #print('Date:', date_time_obj.date())  
        #print('Time:', date_time_obj.time())  
        #print('Date-time:', date_time_obj)  
        bdDomain = preporcess(step[8])
        if bdDomain != '':
            if step1[0] in data:
                data[step1[0]].append((bdDomain))
            else :
                data[step1[0]] = []
                data[step1[0]].append((bdDomain))
            
        
        #duration.total_seconds()  
        
        line = file_object.readline()
    for x, y in data.items():
        print(x, y) 