'''
This program implemented transfer learning on wetlab dataset
'''

sys.path.append('/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/code')
import os.path
import sys
import pandas as pd
import seq2graph
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, random_split
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from warmup_scheduler_pytorch import WarmUpScheduler
import torch_geometric.data
import torch
import torch.nn as nn
from model import GNNModel, GAT, GATv2
from torch.optim import Adam, SGD
import torch_geometric
from torch_geometric.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import networkx as nx
#import wandb
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.nn.init as init

import utils

#from torch_geometric.datasets import TUDataset

print(torch.cuda.is_available())
torch.cuda.empty_cache()
# configure DDP
rank = 0
#SEED = 42
world_size=torch.cuda.device_count()

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

#setup(rank, world_size)

#data = TUDataset(os.path.join('data', 'PROTEINS'), 'PROTEINS')
modelType = 'GATv2'
#modelType = 'GCN'
classificationTask = ['binary', 'multi']
taskList = ['bioReactor', 'benchmark', 'wetlab']
benchmark_dataset_list = ['yeast_chlamydomonas', 'covid_yeast', 'chlamydomonas_zymo', 'covid_chlamydomonas', 'covid_zymo', 'zymo_yeast']
classType = classificationTask[1]
taskType = taskList[2]
#BM_dataset = benchmark_dataset_list[1]
learningMethodList = ['scratch', 'transfer']
learningMethod = learningMethodList[1]

folder_dir = '/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal'

def str2list(input):
    tmp = input[1:-1]
    tmp = tmp.split(', ')
    seq = [int(i) for i in tmp]
    return seq

if taskType == 'bioReactor' and classType == 'binary':
    df = pd.read_csv(f'{folder_dir}/data/raw_signal/combine_ck_ms_signal.csv', sep='\t')
    print(set(df['species']))

    # balance dataset
    df_sub = df[df['species']== 'Clostridium_kluyveri']
    df_sub = df_sub.sample(n=10000, random_state=100)
    df_sub.reset_index(drop=True, inplace=True)
    df_new = pd.concat([df[df['species']=='Methanobrevibacter_smithii'], df_sub])
    df_new = df_new.sample(frac=1)
    df_new.reset_index(drop=True,inplace=True)
    df = df_new

    df['label'] = ''
    df.loc[df['species']=='Methanobrevibacter_smithii', 'label'] = 0
    df.loc[df['species']=='Clostridium_kluyveri', 'label'] = 1
    print(set(df['label']))
    df = df
    df['data'] = list(map(lambda x: str2list(x), df['signal']))
elif taskType == 'bioReactor' and classType == 'multi':
    df = pd.read_pickle(f'{folder_dir}/data/raw_signal/pickle/train/train_five_signal.pkl')
    df = df.rename(columns={'signal': 'data'})
    print(df.shape)
    print(type(df['data'][0]))

    df['label'] = ''
    df.loc[df['species']=='Methanobrevibacter_smithii', 'label'] = 0
    df.loc[df['species']=='Methanothermobacter_thermautotrophicus', 'label'] = 1
    df.loc[df['species']=='Pseudoclavibacter_caeni', 'label'] = 2
    df.loc[df['species']=='Clostridium_ljungdahlii', 'label'] = 3
    df.loc[df['species']=='Clostridium_kluyveri', 'label'] = 4
    print(set(df['label']))
    # one-hot embedding
    #ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit([0,1,2,3,4])
    #label_ = ohe.transform(df['label'])
    #df['label'] = label_
elif taskType == 'benchmark' and classType == 'binary':
    df = pd.read_csv(f'{folder_dir}/data/deepSelectNet/process/{BM_dataset}/train_{BM_dataset}.csv', sep='\t')
    df['label'] = ''
    label_0 = BM_dataset.split('_')[0]
    label_1 = BM_dataset.split('_')[1]
    print(f'Two categories are {label_0} and {label_1}')
    df.loc[df['species']==label_0, 'label'] = 0
    df.loc[df['species']==label_1, 'label'] = 1
    print(set(df['label']))
    df = df
    df = df.rename(columns={'data': 'signal'})
    #df['data'] = list(map(lambda x: str2list(x), df['signal']))
    processed_data = utils.func_multiprocessing(in_seq=df['signal'], func_process=str2list, nThread=64, mode=1, label=None)
    df['data'] = processed_data
elif taskType == 'wetlab' and classType == 'multi':
    df = pd.read_csv(f'{folder_dir}/data/wetlab/train/train_three_signal.csv', sep = '\t')
    df['label'] = ''
    df.loc[df['species']=='cminuta', 'label'] = 1
    df.loc[df['species']=='msmithii', 'label'] = 0
    df.loc[df['species']=='bacteroides_thetaoitamicron', 'label'] = 2
    df = df.rename(columns={'data': 'signal'})
    processed_data = utils.func_multiprocessing(in_seq=df['signal'], func_process=str2list, nThread=64, mode=1, label=None)
    df['data'] = processed_data

df = df[['id', 'data', 'label']]

#if taskType == 'bioReactor' and classType == 'multi':
#    train_set = df
#else:
#    train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=22)
train_set = df


# generate graph data
func_graph = seq2graph.signalGraph()
train_set['graph'] = list(map(lambda x,y: func_graph.graph(sequence=x, label=y), train_set['data'], train_set['label']))

train_set, valid_set = train_test_split(train_set, test_size=0.1, stratify=train_set['label'], random_state=22)

train_set.reset_index(inplace=True, drop=True)
valid_set.reset_index(inplace=True, drop=True)

# hyperparameter

if taskType == 'bioReactor':
    learning_rate = 1e-4
    batch_size = 8
elif taskType == 'wetlab':
    learning_rate = 1e-4
    batch_size = 8
else:
    learning_rate = 1e-5
    batch_size = 32
num_epochs = 256

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.manual_seed(100)

# define dataloader

train_loader = DataLoader(list(train_set['graph']), batch_size=batch_size)
valid_loader = DataLoader(list(train_set['graph']), batch_size=batch_size)

if modelType == 'GAT':
    #model = GAT(n_feat=1, n_class=2, n_layer=2, agg_hidden=64, fc_hidden=64, dropout=0.2, device=device)
    model = GAT(hidden_channels=64, out_channels=2)
elif modelType == 'GATv2' and classType == 'binary':
    #model = GAT(n_feat=1, n_class=2, n_layer=2, agg_hidden=64, fc_hidden=64, dropout=0.2, device=device)
    model = GATv2(hidden_channels=64, out_channels=2)
elif modelType == 'GATv2' and taskType == 'bioReactor' and classType == 'multi':
    model = GATv2(hidden_channels=64, out_channels=5)
elif modelType == 'GATv2' and taskType == 'wetlab' and classType == 'multi':
    model = GATv2(hidden_channels=64, out_channels=3)
elif modelType == 'GCN' and taskType == 'wetlab' and classType == 'multi':
    model = GNNModel(num_features=1, num_classes=3)
elif modelType == 'GCN':
    model = GNNModel(num_features=1, num_classes=2)

# transfer learning
if learningMethod == 'Transfer':
    state_dict = torch.load('/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/bioReactor_model/multi_classification/gatv2/0222/bioReactor_best_model.pth')
    # method 1, random
    state_dict['lin3.weight'] = init.kaiming_normal(torch.randn(3,256), mode='fan_out', nonlinearity='relu')
    state_dict['lin3.bias'] = torch.randn(3)
    model.load_state_dict(state_dict)
    # method2
    #model.load_state_dict(state_dict)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 3)


optimizer = Adam(model.parameters(), lr=learning_rate)
if taskType == 'bioReactor' or taskType == 'wetlab':
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
else:
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
#scheduler_warmup = WarmUpScheduler(optimizer, lr_scheduler, warmup_steps=100, warmup_start_lr=1e-2, warmup_mode='linear')
criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.functional.nll_loss()

class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_metrics(labels, preds, probs, num_class=classType):
    if num_class == 'binary':
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds)
        #mcc = matthews_corrcoef(labels, preds)
        precision = precision_score(labels, preds)
        #print(classification_report(labels, preds))
        #print(confusion_matrix(labels, preds))
        return {
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'auc': auc
            }
    elif num_class == 'multi':
        #auc = roc_auc_score(labels, probs, average='macro', multi_class='ovo')
        f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average='macro')
        #mcc = matthews_corrcoef(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        #print(classification_report(labels, preds))
        #print(confusion_matrix(labels, preds))
        return {
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            #'auc': auc
            }

#wandb.init(project='nano_binaryClassification')
# model training

# configure DDP training
#device = torch.cuda.device(rank)
# initialize PyTorch distributed using environment variables
#dist.init_process_group(backend='nccl', rank=0, world_size=0)
#torch.cuda.set_device(rank)
# set the seed for all GPUs
#torch.cuda.manual_seed_all(SEED)

#model = nn.DataParallel(model)
model = model.to(device)

#model = DistributedDataParallel(model, device_ids=[0,1], output_device=[0])

best_loss = float('inf')
if taskType == 'bioReactor' or taskType == 'wetlab':
    early_stopper = EarlyStopper(patience=20, min_delta=0)
else:
    early_stopper = EarlyStopper(patience=20, min_delta=0)

record_epoch = 0
# save dir
if taskType == 'benchmark':
    save_dir = f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/{taskType}_model/{classType}_classification/gatv2/{BM_dataset}'
elif taskType == 'bioReactor':
    save_dir = f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/{taskType}_model/{classType}_classification/gatv2/0222'
else:
    save_dir = f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/{taskType}_model/{classType}_classification/gatv2/1121'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for epoch in range(num_epochs):
    model.train()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()

    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for data in train_loader:

        data = data.to(device)
        logits = model(data)
        optimizer.zero_grad()
        loss = criterion(logits, data.y)
        #loss = torch.nn.functional.nll_loss(logits, data.y)
        #loss = loss.mean()
        loss.backward()
        optimizer.step()
        #pred = logits.argmax(dim=1)
        #print(pred)
        total_loss += loss.item()

    train_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {train_loss:.4f}")
    
    model.eval()
    #model.to(device)
    all_probs = []
    all_predictions = []
    all_labels = []
    all_loss = 0.0

    with torch.no_grad():

        for data in valid_loader:

            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)
            #loss = torch.nn.functional.nll_loss(logits, data.y)
            #loss = loss.mean()
            prob = logits[:,1]
            _, predictions = torch.max(logits, 1)
            all_loss += loss.item()
            all_probs.extend(prob)
            all_predictions.extend(predictions)
            if classType == 'binary':
                all_labels.extend(data.y)
            elif classType == 'multi':
                #_, true_ = torch.max(data.y, 1)
                #all_labels.extend(true_)
                all_labels.extend(data.y)

    valid_loss = all_loss/len(valid_loader)
    all_labels = torch.Tensor(all_labels).detach().cpu().numpy()
    all_probs = torch.Tensor(all_probs).detach().cpu().numpy()
    all_predictions = torch.Tensor(all_predictions).detach().cpu().numpy()
    
    evaluation_results = compute_metrics(all_labels, all_predictions, all_probs, num_class=classType)

    if classType == 'binary':
        auc = evaluation_results['auc']
    acc = evaluation_results['acc']
    f1 = evaluation_results['f1']
    recall = evaluation_results['recall']
    precision = evaluation_results['precision']

    #scheduler_warmup.step()
    lr_scheduler.step(valid_loss)
    lr = optimizer.param_groups[0]['lr']
    print(f'Learning rate {epoch+1}: {lr}')
    if classType == 'binary':
        print(f'Epoch {epoch+1}: Validation loss: {valid_loss: .4f}, Validation AUC: {auc: .4f}, Validation Accuracy: {acc: .4f}, f1: {f1: .4f}, recall: {recall: .4f}, precision: {precision: .4f}, lr: {lr}')
    elif classType == 'multi':
        print(f'Epoch {epoch+1}: Validation loss: {valid_loss: .4f}, Validation Accuracy: {acc: .4f}, f1: {f1: .4f}, recall: {recall: .4f}, precision: {precision: .4f}, lr: {lr}')

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), f'{save_dir}/{taskType}_best_model.pth')
        if taskType == 'benchmark':
            #torch.save(model.state_dict(), f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/{taskType}_model/binary_classification/gatv2/{BM_dataset}/bioReactor_best_model.pth')
            torch.save(model.state_dict(), f'{save_dir}/{BM_dataset}_best_model.pth')
        else:
            torch.save(model.state_dict(), f'{save_dir}/{taskType}_best_model.pth')
        record_epoch = epoch
#    wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'valid_loss': valid_loss, 'valid_acc': acc, 'valid_f1': f1, 'lr': lr})
    # early stop
    if early_stopper.early_stop(valid_loss):
        print(f'early stopped at epoch {epoch+1}')
        print(f'Saving the last model')
        torch.save(model.state_dict(), f'{save_dir}/ES_{epoch+1}_{valid_loss:.4f}_model.pth')
        break
    #cleanup()

# save the last epoch
torch.save(model.state_dict(), f'{save_dir}/last_{valid_loss:.4f}_model.pth')
print(f'the epoch of the best model {record_epoch+1}')
#wandb.finish()

