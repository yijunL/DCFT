#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import openai
import json
from tqdm import tqdm
import time


# In[2]:


openai.api_key = "xxxx" 

def get_model_list():
    models= openai.Model.list()
    print(models)


def chat(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content":prompt}
    ]
)
    answer = response.choices[0].message.content
    return answer



# In[3]:


# print(get_model_list())




# In[4]:


test_data = json.load(open("test_pubmed_input-5-1.json",'r'))


# In[5]:


print(len(test_data))


# In[6]:


print(openai.api_base)


# In[ ]:

records = {}
try:
    records = json.load(open("records-5-1.json",'r'))
except Exception as e:
    print("error:", e)
    
preds = []
for i_t, i in enumerate(tqdm(test_data)):
    if(str(i_t) in records):
        print(str(i_t)+" Already exist, skip it...")
        continue

    test_sent = " ".join(i["meta_test"]["tokens"])
    test_h = i["meta_test"]["h"][0]
    test_t = i["meta_test"]["t"][0]
    
    meta_train={}
    for i_s, samples in enumerate(i["meta_train"]):
        meta_train[i_s]=[]
        for sample in samples:
            temp = {}
            temp["sent"] = " ".join(sample["tokens"])
            temp["h"] = sample["h"][0]
            temp["t"] = sample["t"][0]
            meta_train[i_s].append(temp)

    prompt=""
    for r in range(len(meta_train)):
        for s_r in meta_train[r]:
            prompt+="As for sentence: "+s_r["sent"]+" The relation between entities " \
            +s_r["h"]+" and "+s_r["t"]+" is R"+str(r)+". "
        
    prompt+="As for sentence: "+test_sent+", the relation between entities " \
        +test_h+" and "+test_t+" is which of the above "+str(len(meta_train))+" relationships? Select the most likely option directly from ("
    for r in range(len(meta_train)):
        prompt+="R"+str(r)
        if(r!=len(meta_train)-1):
            prompt+=','
    prompt+=') without giving a reason:'
    # print(prompt)
    try:
        response = chat(prompt)
        print("response: ",response)

        pr = 0
        for r in range(len(meta_train)):
            if(str(r) in response):
                pr = r
                break
        preds.append(pr)
    
        records[i_t]=pr
        json.dump(records, open("records-5-1.json",'w'), indent=4)
        
    except Exception as e:
        print("error:", e)
        time.sleep(5)


# In[ ]:




