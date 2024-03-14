import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import pickle
import pandas as pd
import random
from time import time
from collections import defaultdict
import warnings
import torch
warnings.filterwarnings('ignore')

n_drugs = 764
n_cells =76 
n_entities = 16232
n_relations = 0
n_nodes = 16232
train_drug_set = defaultdict(list)
test_drug_set = defaultdict(list)
train_dis_set = defaultdict(list)
test_dis_set = defaultdict(list)












def read_triplets(triples):
    global n_entities, n_relations, n_nodes

    #triples = np.loadtxt(file_name, dtype=np.int32)
    #triples = np.unique(triples, axis=0)
    #print(len(triples))

    if args.inverse_r: #测试
        # get triplets with inverse direction like <entity, is-aspect-of, cells>
        inv_triplets_np = triples.copy()
        inv_triplets_np[:, 0] = triples[:, 2]
        inv_triplets_np[:, 2] = triples[:, 0]
        inv_triplets_np[:, 1] = triples[:, 1] + max(triples[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        triples[:, 1] = triples[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((triples, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        triples[:, 1] = triples[:, 1]
        triplets = triples.copy()
    #print(triplets)
    #n_nodes = n_entities=41100

    #n_nodes = n_entities=12015
    
    n_relations = max(triples[:,1])+1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    #for drug1_id,cell_id,drug2_id tqdm(train_data, ascii=True):
        #if [drug1_id,drug2_id] not in rd[0]:
            #rd[0].append([drug1_id,cell_id])
        #if [drug2_id,cell_id] not in rd[0]:
            #rd[0].append([drug2_id,cell_id])


    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(train_data, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            vals = [1.] * len(cf)
            #print(len(vals))
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes)) # row: cf[:,0] # col: cf[:,1] 
        else:
            vals = [1.] * len(np_mat)
            #print(max(np_mat[:,1]))
            #print(max(np_mat[:,0]))
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]


    return adj_mat_list, norm_mat_list, mean_mat_list

def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            #if len(lines)!=4 :continue
            yield lines



def readRecData(path='data_set/OS/comb_final.txt', test_ratio=0.2):
    print('Read Drug Combination Synergy Data...')
    drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, i, r, flod in readTriple(path, sep=','):
        drug_set1.add(int(d1))
        drug_set2.add(int(d2))
        cell_set.add(int(i))
        triples.append((int(d1), int(d2), int(i), int(r), int(flod)))
    
    return list(drug_set1), list(drug_set2), list(cell_set), triples


    



def trans_data(five_data):
    posinter_mat = list()
    neginter_mat = list()
    inter_mat=list()
    for drug_id_1,drug_id_2,cell_id,label,fold_num in five_data:
        if label == 1:
            posinter_mat.append([drug_id_1,cell_id-n_drugs,drug_id_2])
            inter_mat.append([drug_id_1,cell_id-n_drugs,drug_id_2,label])
        else:
            neginter_mat.append([drug_id_1,cell_id-n_drugs,drug_id_2])
            inter_mat.append([drug_id_1,cell_id-n_drugs,drug_id_2,label])
    
    
    return np.array(posinter_mat),np.array(neginter_mat),np.array(inter_mat)

def get_kg(five_data):
    kg_triplets=[]
    for drug_id_1,drug_id_2,cell_id,label,fold_num in five_data:
        if label ==1:
            kg_triplets.append([drug_id_1,cell_id-n_drugs,drug_id_2])
    return kg_triplets




def load_data_Synergy(model_args,i):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    drug1, drug2, cells, triples = readRecData(directory+'comb_final.txt') #读取药物组合数据
    

    #global n_entities, n_relations, n_nodes,n_drugs, n_cells
    
   
    np.random.seed(23)
    #random.shuffle(triples)

    triples_DF = pd.DataFrame(triples)



    idx_test = np.where(triples_DF[4] == i)
    idx_train = np.where(triples_DF[4] != i)
    test_set = [triples[xx] for xx in idx_test[0]]
    #vaild_set=[triples[xx] for xx in idx_test[0][len(idx_test[0])//2:]]
    train_set = [triples[xx] for xx in idx_train[0]]

    com_kg=get_kg(train_set)
    
    #entitys, relations, kgTriples = readKGData(directory+'kg_final2.txt',com_kg)

    train_cf,trainneg_cf,alltrain=trans_data(train_set)
    test_cf,testneg_cf,alltest=trans_data(test_set)

    triples=read_triplets(train_cf)


    #vaild_cf,vaildneg_cf=trans_data(vaild_set)
    #vaild_cf=test_cf[len(test_cf)//2:]
    #test_cf=test_cf[:len(test_cf)//2]
    #vaildneg_cf=testneg_cf[len(testneg_cf)//2:]
    #testneg_cf=testneg_cf[:len(testneg_cf)//2]
    
    
    graph, relation_dict = build_graph(train_cf, triples)

    n_params = {
        'n_drugs': int(n_drugs),
        'n_cells': int(n_cells),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    #drug_dict = {
        #'train_drug_set': train_drug_set,
        #'test_drug_set': test_drug_set,
        #'train_dis_set': train_dis_set,
        #'test_dis_set': test_dis_set,
    #}
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    return directory,train_cf,test_cf,trainneg_cf,testneg_cf, n_params, graph,[adj_mat_list, norm_mat_list, mean_mat_list],alltrain,alltest
           

    



