import time, os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class PclData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(self.len); print(self.data)
        
    def __getitem__(self, index):# function compulsory
        sent = self.data.text[index]
       # sent = " ".join(sent.split())
        inputs = self.tokenizer.encode_plus(
        sent, 
        None,
        add_special_tokens = True,
        max_length=self.max_len,
        #pad_token = '<PAD>',
        padding = 'max_length',
        return_token_type_ids = True,
        truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        #print('debuggin=',ids)
        return { 'ids': torch.tensor(ids, dtype=torch.long),
               'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(self.data.label_[index], dtype=torch.long)
                
               }
    
    def __len__(self):
        return self.len



def load_task1_data(args):

    df = pd.read_csv( args.datafolder + 'dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','label'] )

    df_test_= pd.read_csv( args.datafolder + 'pcl_test.tsv', sep = '\t', names=['seq','id','info','country', 'text','label_'] )

    df_test= df_test_[['text','label_']]
    df_test['label_'] = 1

    df = df.dropna(inplace = False)

    df = df.reset_index(drop = True)


    df_final = df[['text','label']]
    #df_data = data.dropna(subset=['id','info','country'])                 

    #df_final.columns
    df_final['label_']= [ 0 if (y == 1 or y == 0) else 1 for y in df_final['label']]
    df_final.drop('label', axis = 1, inplace= True)

    nclasses = len(list(df_final.label_.unique()))

    my_classes = {c:i for i, c in enumerate(list(df_final.label_.unique()))}
    df_final['label_'] = [my_classes[l] for l in df_final.label_]


    return df_final, df_test