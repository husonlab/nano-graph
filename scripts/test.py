import os.path
import sys
sys.path.append('/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/code')
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
import wandb
import utils
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import time

#from torch_geometric.datasets import TUDataset

print(torch.cuda.is_available())
torch.cuda.empty_cache()

# record time
start_time = time.time()

modelType = 'GATv2'
classificationTask = ['binary', 'multi']
taskList = ['bioReactor', 'benchmark', 'wetlab']
benchmark_dataset_list = ['yeast_chlamydomonas', 'covid_yeast', 'chlamydomonas_zymo', 'covid_chlamydomonas', 'covid_zymo', 'zymo_yeast']
taskType = taskList[2]
classType = classificationTask[1]
BM_dataset = benchmark_dataset_list[1]

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
    df = pd.read_csv(f'{folder_dir}/data/raw_signal/test/test_five_signal.csv', sep='\t')
    df = df.rename(columns={'data': 'signal'})
    print(df.shape)
    print(type(df['signal'][0]))
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
    label_0 = BM_dataset.split('_')[0]
    label_1 = BM_dataset.split('_')[1] 
    df = pd.read_csv(f'{folder_dir}/data/deepSelectNet/process/{BM_dataset}/test_{BM_dataset}.csv', sep='\t')
    df['label'] = ''
    df.loc[df['species']==label_0, 'label'] = 0
    df.loc[df['species']==label_1, 'label'] = 1
    print(set(df['label']))
    print(len(df))
    df = df.rename(columns={'data': 'signal'})
elif taskType == 'wetlab' and classType == 'multi':
    df = pd.read_csv(f'{folder_dir}/data/wetlab/test/test_three_signal.csv', sep = '\t')
    df['label'] = ''
    df.loc[df['species']=='cminuta', 'label'] = 1
    df.loc[df['species']=='msmithii', 'label'] = 0
    df.loc[df['species']=='bacteroides_thetaoitamicron', 'label'] = 2
    df = df.rename(columns={'data': 'signal'})
    processed_data = utils.func_multiprocessing(in_seq=df['signal'], func_process=str2list, nThread=64, mode=1, label=None)
    df['data'] = processed_data



def str2list_multiprocee(in_seq):
    output = []
    for ele in in_seq:
        tmp = ele[1:-1]
        tmp = tmp.split(', ')
        seq = [int(i) for i in tmp]
        output.append(seq)
    return output

processed_data = utils.func_multiprocessing(in_seq=df['signal'], func_process=str2list, nThread=64, mode=1, label=None)
df['data'] = processed_data
print(f'done')
#df['data'] = list(map(lambda x: str2list(x), df['signal']))
#df['data'] = list(map(lambda x: x[0:1000],df['data']))

df = df[['id', 'data', 'label']]

test_set = df


# generate graph data, using multiprocessing
func_graph = seq2graph.signalGraph()
'''
def signalGraph(sequence, label):

    def func_rename(sequence):
        keys = sorted(list(set(sequence)))
        values = list(range(len(keys)))
        name_dict = dict(zip(keys, values))
        new_seq = list(map(lambda x: name_dict[x], sequence))
        return new_seq

    def node(sequence):
        tmp = func_rename(sequence)
        #input = sequence
        #end = np.max(sequence)-np.min(sequence)
        #input = list(range(end+1))
        #tmp = list(range(np.max(sequence)+1))
        node_list = list(set(tmp))
        node_list = torch.tensor([[i] for i in node_list], dtype=torch.float)
        return node_list

    def edge(sequence):
        tmp = func_rename(sequence)
        #input = sequence
        #input = (np.array(sequence)-np.min(sequence)).tolist()
        edge_list_start = tmp[:-1]
        edge_list_end = tmp[1:]
        edge_list = torch.tensor([edge_list_start, edge_list_end], dtype=torch.long)
        return edge_list

    def attr(sequence):
        input = sequence
        #input = (np.array(sequence)-np.min(sequence)).tolist()
        attr_list = []
        for i in range(len(input)-1):
            value = input[i+1]-input[i]
            attr_list.append(value)
        attr_list = torch.tensor(attr_list, dtype=torch.float)
        return attr_list

    graph = Data(x=node(sequence), edge_index=edge(sequence), edge_attr=attr(sequence), y=label)
    return graph
'''

#processed_graph = utils.func_multiprocessing(in_seq=test_set['data'], func_process=signalGraph, nThread=64, mode=2, label=test_set['label'])
#test_set['graph'] = processed_graph
test_set['graph'] = list(map(lambda x,y: func_graph.graph(sequence=x, label=y), test_set['data'], test_set['label']))
print(f'data process done')


# hyperparameter
batch_size = 32
learning_rate = 1e-4
num_epochs = 256

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
torch.manual_seed(100)
# define dataloader

test_loader = DataLoader(list(test_set['graph']), batch_size=batch_size)
print (f'test loader done')
if modelType == 'GAT':
    #model = GAT(n_feat=1, n_class=2, n_layer=2, agg_hidden=64, fc_hidden=64, dropout=0.2, device=device)
    model = GAT(hidden_channels=64, out_channels=2)
elif modelType == 'GATv2' and classType == 'binary':
    #model = GAT(n_feat=1, n_class=2, n_layer=2, agg_hidden=64, fc_hidden=64, dropout=0.2, device=device)
    model = GATv2(hidden_channels=64, out_channels=2)
elif modelType == 'GATv2' and classType == 'multi' and taskType == 'benchmark':
    model = GATv2(hidden_channels=64, out_channels=5)
elif modelType == 'GATv2' and taskType == 'wetlab' and classType == 'multi':
    model = GATv2(hidden_channels=64, out_channels=3)
elif modelType == 'GCN':
    model = GNNModel(num_features=1, num_classes=2)

model = model.to(device)
if taskType == 'bioReactor' and classType == 'binary':
    model.load_state_dict(torch.load('/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/model/binary_classification/gatv2/last_0.4021_model.pth'))
elif taskType == 'bioReactor' and classType == 'multi':
    model.load_state_dict(torch.load('/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/bioReactor_model/multi_classification/gatv2/0222/bioReactor_best_model.pth'))
elif taskType == 'benchmark' and classType == 'binary':
    model.load_state_dict(torch.load(f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/benchmark_model/binary_classification/gatv2/{BM_dataset}/{BM_dataset}_best_model.pth'))
elif taskType == 'wetlab' and classType == 'multi':
    model.load_state_dict(torch.load(f'/ceph/ibmi/ab/projects/wenhuan/project/nanopore_signal/wetlab_model/multi_classification/gatv2/1121/wetlab_best_model.pth'))
print(f'load model done')

def compute_metrics(labels, preds, probs, num_class=classType):
    if num_class == 'binary':
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds)
        #mcc = matthews_corrcoef(labels, preds)
        precision = precision_score(labels, preds)
        #print(classification_report(labels, preds))
        cm = confusion_matrix(labels, preds)
        #print(confusion_matrix(labels, preds))
        return {
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'auc': auc,
            'confusion matrix': cm
            }
    elif num_class == 'multi':
        #auc = roc_auc_score(labels, probs, average='macro', multi_class='ovo')
        f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average='macro')
        #mcc = matthews_corrcoef(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        #print(classification_report(labels, preds))
        cm = confusion_matrix(labels, preds)
        #print(confusion_matrix(labels, preds))
        return {
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'confusion matrix': cm
            #'auc': auc
            }

model = model.to(device)
#model = nn.DataParallel(model)

model.eval()
model.to(device)
all_predictions = []
all_labels = []
all_probs = []
print(f'start training')
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        logits = model(data)
        prob = logits[:,1]
        _, predictions = torch.max(logits, 1)
        all_probs.extend(prob)
        all_predictions.extend(predictions)
        all_labels.extend(data.y)

    #all_predictions = torch.cat(all_predictions, dim=0)
    #all_labels = torch.cat(all_labels, dim=0)
    #evaluation_results = compute_metrics(all_labels.detach().cpu().numpy(), all_predictions.detach().cpu().numpy(), all_probs.detach().cpu().numpy())
    all_labels = torch.Tensor(all_labels).detach().cpu().numpy()
    all_probs = torch.Tensor(all_probs).detach().cpu().numpy()
    all_predictions = torch.Tensor(all_predictions).detach().cpu().numpy()
    print(f'size of samples: {len(all_predictions)}')

    evaluation_results = compute_metrics(all_labels, all_predictions, all_probs, num_class=classType)
    if classType == 'binary':
        auc = evaluation_results['auc']
    acc = evaluation_results['acc']
    f1 = evaluation_results['f1']
    recall = evaluation_results['recall']
    precision = evaluation_results['precision']
    cm = evaluation_results['confusion matrix']

    if classType == 'binary':
        print(f'accuracy {acc: .4f} AUC {auc: .4f} f1 {f1: .4f} precision {precision: .4f} recall {recall: .4f}\nconfusion matrix {cm}')
    else:
        print(f'accuracy {acc: .4f} f1 {f1: .4f} precision {precision: .4f} recall {recall: .4f}\nconfusion matrix {cm}')

print("---total consuming time  %s seconds ---" % (time.time() - start_time))
