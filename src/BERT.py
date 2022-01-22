import torch

from transformers import BertTokenizer, BertModel, CamembertTokenizer, CamembertModel, FlaubertTokenizer, FlaubertModel
import sentencepiece
import pandas as pd
df = pd.read_csv('../Data/dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','label'] )
df = df.dropna(inplace = False)

df = df.reset_index(drop = True)
df.info()

# Importing the libraries needed
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
df_final = df[['text','label']]
#df_data = data.dropna(subset=['id','info','country'])
df_final                  

df_final.shape
#df_final.columns
nclasses = len(list(df_final.text.unique()))
nclasses

#my_classes = {c:i for i, c in enumerate(list(df_final.text.unique()))}
#df_final['label'] = [my_classes[l] for l in df_final.text]

MAX_LEN= 512
TRAIN_BS = 8
VALID_BS = 8
EPOCHS = 1
LR =  1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class CompanyData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):# function compulsory
        corpus = str(self.df.text[index])
        corpus= " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
        text, 
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
                'targets': torch.tensor(self.df.label[index], dtype=torch.long)
                
               }
    
    def __len__(self):
        return self.len
        
train_size = 0.8
train_dataset = df_final.sample(frac= train_size, random_state=20)
test_dataset = df_final.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df_final.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CompanyData(train_dataset, tokenizer, MAX_LEN)
testing_set = CompanyData(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BS,
               'shuffle':True,
               'num_workers': 2}
test_params = {'batch_size': VALID_BS,
              'shuffle': False,
              'num_workers': 2}

trainloader = DataLoader(training_set, **train_params)
testloader = DataLoader(testing_set, **test_params)

class DistillBERTClass(torch.nn.Module):
    def __init__(self, nclasses):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, nclasses)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
model = DistillBERTClass(nclasses)
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =model.parameters(), lr=LR)
def calculate_acc(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/textclassify_experiment_1')

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    model.train()
    for _, data in tqdm(enumerate(trainloader, 0)):
        ids = df['ids'].to(device, dtype=torch.long)
        mask = df['mask'].to(device, dtype=torch.long)
        targets = df['targets'].to(device, dtype=torch.long)
        
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
            print(f'Training Loss per 50 steps: {loss_step}, Training Accuracy: {acc_step}')
            writer.add_scalar('training_loss', loss_step, epoch*len(trainloader) +_)
            
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
    print(f'Total Accuracy Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_acc = (n_correct*100)/nb_tr_examples
    print(f'Training Loss Epoch: {epoch_loss}, Training Accuracy Epoch: {epoch_acc}')
    
    return

# Validation

def valid(model, testloader):
    model.eval()
    n_correct =0; n_wrong =0; total=0
    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            ids = df['ids'].to(device, dtype=torch.long)
            mask = df['mask'].to(device, dtype=torch.long)
            targets = df['targets'].to(device, dtype=torch.long)
            
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_acc(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 100 == 0:
                loss_step = tr_loss/nb_tr_steps
                acc_step = (n_correct*100)/nb_tr_examples
                print(f'Validation Loss per 100 steps: {loss_step}, Validation Accuracy per: {acc_step}')   
        
    epoch_loss = tr_loss/nb_tr_steps
    epoch_acc = (n_correct*100)/nb_tr_examples
    print(f'Validation Loss Epoch: {epoch_loss}, Validation Accuracy Epoch: {epoch_acc}')
    
    return epoch_acc

for epoch in range(EPOCHS):
    train(epoch)
    
acc = valid(model, testloader)