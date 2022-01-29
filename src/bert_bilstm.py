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


from utils import EarlyStopping
from data import PclData, load_task1_data
#from BI_LSTM import BiLSTM

def main(args):

    df_final, df_test = load_task1_data(args)

    nclasses = len(list(df_final.label_.unique()))
    my_classes = {c:i for i, c in enumerate(list(df_final.label_.unique()))}

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


    hdimdict = {'xlm-roberta-base': 250002,
    'distilbert-base-uncased': 30522,
    'gpt2': 768, 'bert-base-uncased': 30522,
    'roberta-large': 768,
    }


    print(f'{args.lm} is running.......')

    device = 'cuda' if cuda.is_available() else 'cpu'



    train_size = 0.8
    train_dataset = df_final.sample(frac= train_size, random_state=42)
    val_dataset = df_final.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = df_test

    print("FULL Dataset: {}".format(df_final.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = PclData(train_dataset, tokenizer, MAX_LEN)
    val_set = PclData(val_dataset, tokenizer, MAX_LEN)
    test_set= PclData(test_dataset, tokenizer, MAX_LEN)

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
                    'input_dim':hdimdict[args.lm],
                    'bs': args.bs, 
                    'seq_len': MAX_LEN
                    }

    class BiLSTM(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, n_layers=1, bidir=True, bs=32):
            super(BiLSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidir)
            self.d =  2 if bidir else 1
            
            self.hidden = (torch.zeros(self.n_layers*self.d,1*bs,self.hidden_dim),
                                torch.zeros(self.n_layers*self.d,1*bs,self.hidden_dim))
            
        def forward(self, x):
            out, self.hidden = self.lstm(x, self.hidden)
            return out

    class BertBilstmModelClass (torch.nn.Module):
        def __init__(self, lmodel, input_dim,bs, seq_len, nclasses):
            super(BertBilstmModelClass, self).__init__()
            self.seq_len = seq_len
            self.bs = bs
            self.l1 = lmodel
            self.pre_classifier = torch.nn.Linear(self.seq_len*2*100, 64)
            self.dropout = torch.nn.Dropout(0.3)
            self.bilstm = BiLSTM(input_dim=input_dim,bs=bs)
            self.classifier = torch.nn.Linear(768, nclasses)

        def forward(self, input_ids, attention_mask):
            with torch.autograd.no_grad():
                output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            
            lstmop = self.bilstm(hidden_state)
            pooler = self.pre_classifier(lstmop.reshape(self.bs, -1))
            pooler = torch.nn.ReLU()(pooler)
            pooler = self.dropout(pooler)
            output = self.classifier(pooler)
            return output

    model = BertBilstmModelClass(**model_config)
    model.to(device)
    print(model)

    wt_array =len(df_final['text'])/(len(set(df_final['label_']))*(np.bincount(df_final['label_'])))
    wt_array

    class_weights=torch.FloatTensor(wt_array).to(device=device)

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
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
            #print(ids.shape, mask.shape)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_acc(big_idx, targets)
            
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 50 == 0:
                
                loss_step = tr_loss/nb_tr_steps
                acc_step = (n_correct*100)/nb_tr_examples
                #print(f'Training Loss per 50 steps: {loss_step}, Training Accuracy: {acc_step}')
                #writer.add_scalar('training_loss', loss_step, epoch*len(trainloader) +_)
                
                
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
       # print(f'Total Accuracy Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_acc = (n_correct*100)/nb_tr_examples
        print(f'Epoch : {epoch}, training Loss Epoch: {epoch_loss}, Training Accuracy Epoch: {epoch_acc}')
        acc, y_true, y_pred,vloss = valid(model, valloader)
        print(classification_report(y_true, y_pred))
        early_stopping(vloss, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            return
        return

    # Validation

    def valid(model, testloader):
        model.eval()
        n_correct =0; n_wrong =0; total=0; tr_loss =0; nb_tr_steps =0; nb_tr_examples=0
        y_pred, y_true = [],[]
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                
                outputs = model(ids, mask)
                loss = loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += calculate_acc(big_idx, targets)

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

    acc, y_true, y_pred, _  = valid(model, testloader)


    
    accuracy = confusion_matrix(y_true, y_pred)
    print(accuracy)
    
    print(classification_report(y_true, y_pred))

    
    f =open(args.loglocation + f'{args.lm}_bilstm_test_1.txt', 'w')
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

    