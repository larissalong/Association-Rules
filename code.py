#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:44:43 2020

@author: qingqinglong
"""

#!pip install apyori
#!pip install requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
import requests
from io import StringIO


# import data
r = requests.get('https://drive.google.com/uc?authuser=0&id=1y5DYn0dGoSbC22xowBq2d4po6h1JxcTQ&output=csv')
data = r.content

# convert to dataframe
s=str(data,'utf-8')
data2 = StringIO(s) 
df=pd.read_csv(data2, header=None)

# convert to list
records = df.stack().groupby(level=0).apply(list).tolist()
records[:10]

# apply algorithm
association_rules = apriori(records, min_support=0.01, min_confidence=0.1, 
                            min_lift=1, min_length=2, max_length=2)
association_results = list(association_rules)

# view results
print(len(association_results))
association_results

# save results
res = []

for item in association_results:
    # first index of the inner list, contains base item and added item
    pair = item[0] 
    items = [x for x in pair]
    
    # look at those transactions with two categories
    if len(items) > 1:
    
        item1 = str(items[0])
        item2 = str(items[1])
    
        # second index
        support = str(item[1])
    
        #third index
        confidence = str(item[2][0][2])
        lift = str(item[2][0][3])
    
        res.append((item1, item2, support, confidence, lift))
    
# convert result to dataframe
table = pd.DataFrame(res)
table.rename(columns = {0:'Base Item', 1:'Attached Item', 2:'Support', 
                        3:'Confidence', 4:'Lift'}, 
             inplace = True)
table[['Support', 'Confidence', 'Lift']] = table[['Support', 'Confidence', 'Lift']].astype('float32')
table['Frequency'] = round(table['Support'] * len(data))

# view results
table.head()
