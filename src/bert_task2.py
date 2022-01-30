#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import cuda
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


from utils import EarlyStopping
from data import load_task2_data
#from BI_LSTM import BiLSTM

def main(args):

    df_final, y, df_test = load_task2_data(args)

    nclasses = 7#len(list(df_final.label_.unique()))
    #my_classes = {c:i for i, c in enumerate(list(df_final.label_.unique()))}

    MAX_LEN= 512
    TRAIN_BS = args.bs
    VALID_BS = args.bs
    EPOCHS = args.epochs
    LR =  1e-05

    if args.lm != 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(args.lm)
        lmodel = AutoModelForMaskedLM.from_pretrained(args.lm)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.lm)
        lmodel = GPT2Model.from_pretrained(args.lm)
        tokenizer.pad_token = tokenizer.eos_token


    hdimdict = {'xlm-roberta-base': 250002,
    'distilbert-base-uncased': 30522,
    'gpt2': 768, 'bert-base-uncased': 30522,
    'roberta-large': 50265,
    }


    print(f'{args.lm} is running.......')

    device = 'cuda' if cuda.is_available() else 'cpu'

    class PclData(Dataset):
        def __init__(self, dataframe, targets, tokenizer, max_len):
            self.len = len(dataframe)
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.targets = targets
            #print(self.len)#; print(self.data)
            
        def __getitem__(self, index):# function compulsory
            sent = self.data.paragraph[index]
           # sent = " ".join(sent.split())
            inputs = self.tokenizer.encode_plus(
            sent, 
            None,
            add_special_tokens = True,
            max_length=self.max_len,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation=True
            )
            
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            
            return { 'ids': torch.tensor(ids, dtype=torch.long),
                   'mask': torch.tensor(mask, dtype=torch.long),
                    'targets': torch.tensor(self.targets[index], dtype=torch.long)
                    
                   }
        
        def __len__(self):
            return self.len



    X_data= df_final.drop_duplicates(subset=['paragraph'])#df_final['paragraph'].unique()
    #print(type(X_data))
    train_dataset, val_dataset, y_train, y_val = train_test_split(X_data,  y, test_size = 0.20, random_state= 0)
    train_dataset= train_dataset.reset_index(drop=True)
    val_dataset= val_dataset.reset_index(drop=True)
    df_test = df_test.rename(columns={"text": "paragraph"})
    test_dataset = df_test

    #print(test_dataset)
    y_test = np.random.randint(2, size=(len(df_test),7))

    print("FULL Dataset: {}".format(df_final.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = PclData(train_dataset, y_train, tokenizer, MAX_LEN)
    val_set = PclData(val_dataset, y_val, tokenizer, MAX_LEN)
    test_set= PclData(test_dataset, y_test, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BS,
                   'shuffle':True,
                   'num_workers': 2}
    val_params = {'batch_size': VALID_BS,
                  'shuffle': False,
                  'num_workers': 2}

    trainloader = DataLoader(training_set, **train_params)
    valloader = DataLoader(val_set, **val_params)
    testloader = DataLoader(test_set, **val_params)

    
    
    model_config = {'lmodel': lmodel,
                    'nclasses': nclasses,
                    'hidden_dim':hdimdict[args.lm],
                    }

    class RobertaModelClass(torch.nn.Module):
        def __init__(self, lmodel, hidden_dim, nclasses):
            super(RobertaModelClass, self).__init__()
            self.l1 = lmodel#RobertaModel.from_pretrained('roberta-base')
            self.pre_classifier = torch.nn.Linear(hidden_dim, 768)
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(768, nclasses)

        def forward(self, input_ids, attention_mask):
            with torch.autograd.no_grad():
                output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            pooler = self.pre_classifier(pooler)
            pooler = torch.nn.ReLU()(pooler)
            pooler = self.dropout(pooler)
            output = self.classifier(pooler)
            return output


    model = RobertaModelClass(**model_config)
    model.to(device)
    #print(model)

    wt_array = len(X_data)/(nclasses*np.sum(y, axis=0))
    wt_array

    class_weights=torch.FloatTensor(wt_array).to(device=device)

    loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    optimizer = torch.optim.Adam(params =model.parameters(), lr=LR)
    def calculate_acc(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct


    early_stopping = EarlyStopping(patience=5, verbose=True)

    #writer = SummaryWriter('runs/textclassify_experiment_1')

    def train(epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        
        model.train()
        for _, data in tqdm(enumerate(trainloader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            
            outputs = model(ids, mask)
            loss = loss_function(outputs.float(), targets.float())
            tr_loss += loss.item()
            big_val= (torch.sigmoid(outputs.data)>.5).float()
            #print(big_val, targets)
            n_correct += accuracy_score(big_val.detach().cpu().numpy(), targets.detach().cpu().numpy())
            
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 50 == 0:
                
                loss_step = tr_loss/nb_tr_steps
                acc_step = (n_correct*100)/nb_tr_examples
                print(f'Training Loss per 50 steps: {loss_step}, Training Accuracy: {acc_step}')
                #writer.add_scalar('training_loss', loss_step, epoch*len(trainloader) +_)
                
                
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
       # print(f'Total Accuracy Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_acc = (n_correct*100)/nb_tr_examples
        print(f'Epoch : {epoch}, training Loss Epoch: {epoch_loss}, Training Accuracy Epoch: {epoch_acc}')
        acc, y_true, y_pred,vloss = valid(model, valloader)
        #print(classification_report(y_true, y_pred))
        early_stopping(vloss, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            return
        return

    # Validation

    def valid(model, valloader):
        model.eval()
        n_correct =0; n_wrong =0; total=0; tr_loss =0; nb_tr_steps =0; nb_tr_examples=0
        y_pred, y_true = [],[]
        with torch.no_grad():
            for _, data in enumerate(valloader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                
                outputs = model(ids, mask)
                loss = loss_function(outputs.float(), targets.float())
                tr_loss += loss.item()
                big_val= (torch.sigmoid(outputs.data)>.5).float()
                n_correct += accuracy_score(big_val.detach().cpu().numpy(), targets.detach().cpu().numpy())

                y_true.extend(targets.cpu().detach().numpy())
          
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())
                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)
                
                if _ % 100 == 0:
                    loss_step = tr_loss/nb_tr_steps
                    acc_step = (n_correct*100)/nb_tr_examples
                    #print(f'Validation Loss per 100 steps: {loss_step}, Validation Accuracy per: {acc_step}')   
            
        epoch_loss = tr_loss/nb_tr_steps
        epoch_acc = (n_correct*100)/nb_tr_examples
        print(f'Validation Loss Epoch: {epoch_loss}, Validation Accuracy Epoch: {epoch_acc}')
        
        return epoch_acc, y_true, y_pred, epoch_loss

    for epoch in range(EPOCHS):
        train(epoch)
    print("------Testing---------")
    acc, y_true, y_pred, _  = valid(model, testloader)


    
    #accuracy = accuracy_score(y_true, y_pred)
    #print(accuracy)
    
    #print(classification_report(y_true, y_pred))

    
    f =open(args.loglocation + f'{args.lm}_demo_test_task2_1.txt', 'w')
    for i in y_pred:
      print(i, file = f )
    f.close()
    print('file saved!!!!')

if __name__ == '__main__':

    import argparse, pathlib

    parser = argparse.ArgumentParser(description="Running bert+bilstm...")
    
    parser.add_argument('--datafolder', default = '~/PCL/Data/', 
                        help='Location to features folder')
    parser.add_argument('--loglocation', default= '~/PCL/logs/',
                        help='Location to save logs')
    parser.add_argument('--epochs', default =20, type=int,
                                                help='Number of epochs')
    parser.add_argument('--bs', default =32, type=int,
                                                help='Batch size')
    parser.add_argument('--lm', default='bert-base-uncased', required=True,
                        help='Language Model Name')

    args = parser.parse_args()

    main(args)

    
