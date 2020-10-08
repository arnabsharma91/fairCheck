#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv


# In[3]:



def funcWriteXml(df):
    
    f = open('dataInput.xml', 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?> \n <Inputs> \n')
    
    for i in range(0, df.shape[1]):
        f.write('<Input> \n <Feature-name>')
        f.write(df.columns.values[i])
        f.write('<\Feature-name> \n <Feature-type>')
        f.write(str(df.dtypes[i]))
        f.write('<\Feature-type> \n <Value> \n <minVal>')
        f.write(str(df.iloc[:, i].min()))
        f.write('<\minVal> \n <maxVal>')
        f.write(str(df.iloc[:, i].max()))
        f.write('<\maxVal> \n <\Value> \n <\Input>\n')
    
    f.write('<\Inputs>') 
    f.close()


# In[4]:


df = pd.read_csv('Datasets/Adult.csv')
funcWriteXml(df)


# In[13]:


fe_dict = {}
for i in range(0, df.shape[1]):
    fe_dict[df.columns.values[i]] = str(df.dtypes[i])


# In[17]:


try:
    with open('feNameType.csv', 'w') as csv_file:
        writer = cv.writer(csv_file)
        for key, value in fe_dict.items():
            writer.writerow([key, value])
except IOError:
    print("I/O error")


# In[ ]:





# In[ ]:




