
from operator import neg
import random
import copy
import torch
import numpy as np
import math
import torch.nn as nn

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data_Synergy
from modules.KGIN import Recommender,DNN
from modules.Multiview import MultiViewNet
from utils.evaluate import result_test,result_train
from utils.metrics import AUC,AUPR,ACC
from utils.helper import early_stopping
import collections
import pandas as pd
from utils.evaluate import sigmoid_function
from modules.l0dense import L0Dense
from modules.encoder import encoder
from modules.aggregator import aggregator

from sklearn import metrics

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

n_drugs = 0
n_diseases = 0
n_entities = 0
n_nodes = 0
n_relations = 0



#descriptors preparation


def get_feed_dict(train_entity_pairs, start, end,trainneg_drug_set):
    def get_negative(drug_disease,trainneg_drug_set):
        neg_diseases = []
        for drug, _ in drug_disease.cpu().numpy():
            drug = int(drug)
            while True:
                #print(trainneg_drug_set[drug])
                neg_disease = np.random.choice(trainneg_drug_set[drug])
                break
            neg_diseases.append(neg_disease)
        return neg_diseases

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['drugs'] = entity_pairs[:, 0]
    feed_dict['pos_diseases'] = entity_pairs[:, 1]
    feed_dict['neg_diseases'] = torch.LongTensor(get_negative(entity_pairs,trainneg_drug_set)).to(device) #获取负样本
    return feed_dict

def new_get_batch(train_entity_pairs_pos,start,end,train_entity_pairs_neg,batch_size):
    feed_dict = {}
    entity_pairs = train_entity_pairs_pos[start:end].to(device)
    feed_dict['drugs1'] = entity_pairs[:, 0]
    feed_dict['drugs2'] = entity_pairs[:, 2]
    feed_dict['pos_cell'] = entity_pairs[:, 1]
    neg_index = np.random.randint(len(train_entity_pairs_neg), size=batch_size)
    #print(len(neg_index))
    neg_entity_pairs=train_entity_pairs_neg[neg_index,].to(device)
    #print(neg_entity_pairs)
    feed_dict['drugs1']=torch.cat([feed_dict['drugs1'],neg_entity_pairs[:,0]],0)
    feed_dict['drugs2']=torch.cat([feed_dict['drugs2'],neg_entity_pairs[:,2]],0)
    feed_dict['neg_cell']=neg_entity_pairs[:,1]
    feed_dict['cells']=torch.cat([feed_dict['pos_cell'],feed_dict['neg_cell']],0)
    feed_dict['labels']=torch.tensor(np.concatenate((np.ones(batch_size),np.zeros(batch_size)))).to(device)
    #print(feed_dict['drugs'],len(feed_dict['drugs']))
    #print(feed_dict['pos_diseases'],len(feed_dict['pos_diseases']))
    #print(feed_dict['neg_diseases'],len(feed_dict['neg_diseases']))
    return feed_dict

def new_new_get_batch(cf,start,end,batch_size):
    feed_dict = {}
    entity_pairs = cf[start:end].to(device)
    drug1=alltraincf[:,0][start:end]
    drug2=alltraincf[:,2][start:end]
    cell=alltraincf[:,1][start:end]
    label=alltraincf[:,3][start:end]

    #print(feed_dict['drugs'],len(feed_dict['drugs']))
    #print(feed_dict['pos_diseases'],len(feed_dict['pos_diseases']))
    #print(feed_dict['neg_diseases'],len(feed_dict['neg_diseases']))
    return drug1,drug2,cell,label

def get_sim_edge(sim,threshold):
    col=[]
    row=[]
    for i in range(len(sim)):
        for j in range(len(sim[0])):
            if sim[i][j]> threshold:
                row.append(i)
                col.append(j)
    return([col,row])





def KGIN(i):
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device,n_diseases,n_drugs
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    print(torch.cuda.is_available())
    print(device)
    
    """build dataset"""
    datapath,train_cf, test_cf,trainneg_cf, testneg_cf ,n_params, graph,mat_list,alltrain,alltest = load_data_Synergy(args,i) #读取数据
    #drugSimedge=get_sim_edge(drugSimfeat,0.3)
    #disSimedge=get_sim_edge(diseasesimfeat,0.1)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    
    
    n_drugs = n_params['n_drugs']
    
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    
    #print(n_drugs,n_diseases,n_entities,n_relations,n_nodes)
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1],cf[2]] for cf in train_cf], np.int32))
    train_cf_pairs_neg = torch.LongTensor(np.array([[cf[0], cf[1],cf[2]] for cf in trainneg_cf], np.int32))

    #ufeature2 = {}
    #ifeature2 = {}
    #for i in range(n_drugs):
        #ufeature2[i] = [0 for _ in range(n_drugs)]
    #for i in range(267,n_diseases):
        #ifeature2[i] = [0 for _ in range(n_diseases)]
    
    #for key in drug_dict['train_drug_set'].keys():
        #ufeature2[key] = drugSimfeat[key].tolist()
    #for key in drug_dict['train_dis_set'].keys():
        #ifeature2[key] = diseasesimfeat[key-267].tolist()



    #drugfea2 = []
    #for key in ufeature2.keys():
        #drugfea2.append(ufeature2[key])
    #drugfea2 = torch.Tensor(np.array(drugfea2, dtype=np.float32))
    #drugemb2 = nn.Embedding(n_drugs, n_drugs)
    #drugemb2.weight = torch.nn.Parameter(drugfea2)
    #drugfea2=drugemb2(drugfea2)

    #diseasefea2 = []
    #for key in ifeature2.keys():
        #diseasefea2.append(ifeature2[key])
    #diseasefea2 = torch.Tensor(np.array(diseasefea2, dtype=np.float32))
    #disemb2 = nn.Embedding(n_diseases,n_diseases)
    #disemb2.weight = torch.nn.Parameter(diseasefea2)

    #print(drugemb2)

    #u_agg_embed_cmp2 = aggregator(drugemb2.to(device), disemb2.to(device), drug_dict['train_drug_set'], 64, cuda=device,
                                  #weight_decay=0.0005, droprate=0.5).to(device)

    #u_embed_cmp2 = encoder(64, u_agg_embed_cmp2, cuda=device)
    

    #i_agg_embed_cmp2 = aggregator(drugemb2.to(device), disemb2.to(device), drug_dict['train_dis_set'], 64, cuda=device,
                                  #weight_decay=0.0005, droprate=0.5, is_drug_part=False).to(device)
    #i_embed_cmp2 = encoder(64, i_agg_embed_cmp2, cuda=device, is_user_part=False)





    
    """define model"""
    #mean_mat_list_drug = mean_mat_list[0].tocsr()[:n_drugs, :].tocoo() # 整个矩阵 n_nodes * n_nodes 现在：n_drugs* n_nodes
    #print(type(mean_mat_list[0].tocsr()[:, n_drugs:n_diseases+n_drugs]))
    #print(type(mean_mat_list[0].T))
    #mean_mat_list_disease = mean_mat_list[0].T.tocsr()[n_drugs:n_diseases+n_drugs, :].tocoo() # n_diseases* n_nodes 
    #print(mean_mat_list_disease)
    model = Recommender(n_params, args, graph).to(device)
    

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_model=None
    best_train = [0,0,0]
    best_test = [0,0,0]
    stopping_step = 0
    should_stop = False
    best_loss=100
    
    print("start knowledge graph training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        
        #useddata=[]
        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            #batch = get_feed_dict(train_cf_pairs,
                            #s, s + args.batch_size,
                            #drug_dict['trainneg_user_set'])
            batch = new_get_batch(train_cf_pairs,
                            s, s + args.batch_size,
                            train_cf_pairs_neg,args.batch_size)
            

            batch_loss, _, _ = model(batch)
            #print(batch_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()
        
        if epoch % 10 == 9 or epoch == 1:
            """testing for test dataset"""
            test_s_t = time()
            #train_auc,train_aupr,train_acc = result_train(model,train_cf,trainneg_cf)
            test_auc,test_aupr,test_acc,test_pred = result_test(model,alltest)
            train_auc,train_aupr,train_acc,train_pred = result_train(model,alltrain)
            #train_result=[train_auc,train_aupr,train_acc]
            test_result=[test_auc,test_aupr,test_acc]
            train_result=[train_auc,train_aupr,train_acc]
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss","train_AUC","train_AUPR","train_ACC", "test_AUC","test_AUPR","test_ACC"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), train_auc,train_aupr,train_acc,test_auc,test_aupr,test_acc]
            )
            print(train_res)
            
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            best_train,best_test, best_loss,stopping_step, should_stop = early_stopping(best_loss,loss.item(),best_train,best_test,train_result,test_result,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=10)
            if stopping_step==0:
                best_pred_test=test_pred
                best_pred_train=train_pred
            
            if should_stop:
                break

            """save weight"""
            

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.disease()))
            print('using time %.4f, training loss at epoch %d: %.4f ' %(train_e_t - train_s_t, epoch, loss))
    

    drug_emb,cell_emb=model.get_ent_rel_emb()
    pd.DataFrame(drug_emb.detach().cpu()).to_csv( 'OSmodel/data/KGANSynergy/OS/CG256/embeddings_drugs_G2_'+str(i+1)+'.txt',sep='\t', header=None, index=True)
    pd.DataFrame(cell_emb.detach().cpu()).to_csv('OSmodel/data/KGANSynergy/OS/CG256/embeddings_cells_G2_'+str(i+1)+'.txt',sep='\t', header=None, index=True)
    print('early stopping at %d, train AUC:%.4f, train AUPR:%.4f,train ACC: %.4f,  Test AUC :%.4f, Test AUPR: %.4f,  Test ACC: %.4f' % (epoch, best_train[0],best_train[1],best_train[2],best_test[0],best_test[1],best_test[2]))
    
    return(best_pred_test,best_pred_train,best_train,best_test,model,alltrain,alltest,device)

def bpr_loss(scores,labels):
    pos_scores=[]
    neg_scores=[]
    indx=len(labels)
    for i in range(indx):
        if labels[i]==0:
            neg_scores.append(scores[i])
        else:
            pos_scores.append(scores[i])
    pos_scores=torch.stack(pos_scores, 0)
    neg_scores=torch.stack(neg_scores, 0)
    if len(pos_scores)>=len(neg_scores):
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores[:len(neg_scores)] - neg_scores))# BRP loss
    else:
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores[:len(pos_scores)]))

    return mf_loss
def sigmoid_function(z):
    fz=[]
    for num in z:
        fz.append(1/(1+math.exp(-num)))
    return fz




if __name__=='__main__':
    Train=[]
    
    TestAUC=[]
    TestAUPR=[]
    TestACC=[]
    TestnewAUC=[]
    TestnewAUPR=[]
    TestnewACC=[]
    for i in range(1,6):
        print(i)
        CG_pred1,CG_pred2,train,test,kg_model,alltraincf,alltestcf,device=KGIN(i-1)
        #np.save('OSmodel/splitresult/train/CG'+str(i)+'.npy',CG_pred2)
        #np.save('OSmodel/splitresult/test/CG'+str(i)+'.npy',CG_pred1)
        TestAUC.append(test[0])
        TestAUPR.append(test[1])
        TestACC.append(test[2])

        print("DNN")

        #drug_emb2,cell_emb2=kg_model.get_ent_rel_emb()
        drug_emb2 = pd.read_csv('OSmodel/data/KGANSynergy/OS/CG256/embeddings_drugs_G2_'+str(i)+'.txt',sep='\t', header=None, index_col=0)
        cell_emb2 = pd.read_csv('OSmodel/data/KGANSynergy/OS/CG256/embeddings_cells_G2_'+str(i)+'.txt',sep='\t', header=None, index_col=0)
        drug_emb2=torch.tensor(np.float32(drug_emb2[:n_drugs])).to(device)
        cell_emb2=torch.tensor(np.float32(cell_emb2)).to(device)
        
        

        all_emb1=np.loadtxt('OSmodel/data/KGANSynergy/OS/256/entity_emb_TransE_os_256.txt',delimiter='\t')
        drug_emb1=torch.tensor(np.float32(all_emb1[:n_drugs])).to(device)
        cell_emb1=torch.tensor(np.float32(all_emb1[13043:13043+29])).to(device)
        #drug_emb1=torch.tensor(np.float32(all_emb1[:21])).to(device)

        drug_smiles=np.loadtxt("OSmodel/data/KGANSynergy/OS//DrugMACCS.txt")
        drug_smiles=torch.tensor(np.float32(drug_smiles)).to(device)
        cell_express=np.loadtxt("OSmodel/data/KGANSynergy/OS/cell_expression.txt",delimiter='\t')
        cell_express=torch.tensor(np.float32(cell_express)).to(device)
        #print(cell_express.size())
        
        
      

        batch_size=32
        alltraincf = torch.LongTensor(np.array([[cf[0], cf[1], cf[2], cf[3]] for cf in alltraincf], np.int32))
        alltestcf = torch.LongTensor(np.array([[cf[0], cf[1], cf[2], cf[3]] for cf in alltestcf], np.int32))
       


        #model2=MultiViewNet(1).to(device)
        #model2=DNN(args.dim,args.dim,args.dim,args.dim,1024).to(device)
        model2=DNN(args.dim,args.dim,args.dim,args.dim,2048).to(device)
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
        loss_func = nn.BCELoss()

        best_train = [0,0,0]
        best_test = [0,0,0]
        stopping_step = 0
        should_stop = False
        best_loss=100

        for epoch in range(1000):
            index=np.arange(len(alltraincf))
            np.random.shuffle(index)
            alltraincf=alltraincf[index]

            loss,s,cor_loss=0,0,0
            while s+batch_size<=len(alltraincf):
                drug1,drug2,cell,label=new_new_get_batch(alltraincf,s,s+batch_size,batch_size)
                drug1_feats1=drug_emb1[drug1]
                drug1_feats2=drug_emb2[drug1]
                smile_drug1=drug_smiles[drug1]
                drug2_feats1=drug_emb1[drug2]
                drug2_feats2=drug_emb2[drug2]
                smile_drug2=drug_smiles[drug2]
                cell_feats=cell_emb1[cell]
                cell_feats2=cell_emb2[cell]
                cell_3=cell_express[cell]
                y_true=label.to(device)
                y_pred = model2(drug1_feats1, drug1_feats2,smile_drug1, drug2_feats1, drug2_feats2,smile_drug2 ,cell_feats, cell_feats2,cell_3)
                batch_loss = loss_func(y_pred.float().squeeze(), y_true.float())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss+=batch_loss
                s+=args.batch_size
            
            if epoch % 10 == 9 or epoch == 1:
                
                drug1_feats1=drug_emb1[alltraincf[:,0]]
                drug1_feats2=drug_emb2[alltraincf[:,0]]
                smile_drug1=drug_smiles[alltraincf[:,0]]
                drug2_feats1=drug_emb1[alltraincf[:,2]]
                drug2_feats2=drug_emb2[alltraincf[:,2]]
                smile_drug2=drug_smiles[alltraincf[:,2]]
                cell_feats=cell_emb1[alltraincf[:,1]]
                cell_feats2=cell_emb2[alltraincf[:,1]]
                cell_3=cell_express[alltraincf[:,1]]
                label=alltraincf[:,3]
                y_pred = model2(drug1_feats1, drug1_feats2,smile_drug1, drug2_feats1, drug2_feats2, smile_drug2,cell_feats, cell_feats2,cell_3).detach().cpu().squeeze()
                train_auc=AUC(label,y_pred)
                train_aupr=AUPR(label,y_pred)
                train_acc=ACC(label,y_pred)
                

                drug1_feats1=drug_emb1[alltestcf[:,0]]
                drug1_feats2=drug_emb2[alltestcf[:,0]]
                smile_drug1=drug_smiles[alltestcf[:,0]]
                drug2_feats1=drug_emb1[alltestcf[:,2]]
                drug2_feats2=drug_emb2[alltestcf[:,2]]
                smile_drug2=drug_smiles[alltestcf[:,2]]
                cell_feats=cell_emb1[alltestcf[:,1]]
                cell_feats2=cell_emb2[alltestcf[:,1]]
                cell_3=cell_express[alltestcf[:,1]]
                label=alltestcf[:,3]

                y_pred = model2(drug1_feats1, drug1_feats2,smile_drug1 ,drug2_feats1, drug2_feats2,smile_drug2, cell_feats, cell_feats2,cell_3).detach().cpu().squeeze()
                test_auc=AUC(label,y_pred)
                test_aupr=AUPR(label,y_pred)
                test_acc=ACC(label,y_pred)

                test_result=[test_auc,test_aupr,test_acc]
                train_result=[train_auc,train_aupr,train_acc]
                

                

                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "Loss","train_AUC","train_AUPR","train_ACC", "test_AUC","test_AUPR","test_ACC"]
                train_res.add_row(
                    [epoch, loss.item(), train_auc,train_aupr,train_acc,test_auc,test_aupr,test_acc]
                )
                print(train_res)

                best_train,best_test, best_loss,stopping_step, should_stop = early_stopping(best_loss,loss.item(),best_train,best_test,train_result,test_result,
                                                                    stopping_step, expected_order='dcc',
                                                                    flag_step=5)
                
                if should_stop:
                    break
            else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.target()))
                print(' training loss at epoch %d: %.4f, cor: %.6f' % ( epoch, loss, cor_loss))
        TestnewAUC.append(best_test[0])
        TestnewAUPR.append(best_test[1])
        TestnewACC.append(best_test[2])
    print(1)

    #test_pos_score=[]
    #test_neg_score=[]
    #savemodel,AUPR1,AUC1,test_cf,testneg_cf=KGIN(i)
    #entity_gcn_emb,drug_gcn_emb=savemodel.generate()
    #for u_id,i_id in test_cf:
        #test_pos_score.append(savemodel.rating(drug_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    #for u_id,i_id in testneg_cf:
        #test_neg_score.append(savemodel.rating(drug_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    
    #test_pos_score.extend(test_neg_score)
    #test_pos_score=sigmoid_function(test_pos_score)
    #fw1=open('data/Y08/warm_start_1_10/test_result_'+str(i)+'.txt','w')
    #fw1=open('data/luo/protein_coldstart/test_result_'+str(i)+'.txt','w')
    #for score in test_pos_score:
        #fw1.write(str(score))
        #fw1.write('\n')
    #print(truthpos,test_pos_score)
    #AUPR.append(AUPR1)
    #AUC.append(AUC1)
    #fw1=open('OSmodel/result/os/test_result_model_G2.txt','w')
    #for i in range(5):
        #fw1.write(str(TestAUC[i]))
        #fw1.write(',')
    #fw1.write('\n')
    #for i in range(5):
        #fw1.write(str(TestAUPR[i]))
        #fw1.write(',')
    #fw1.write('\n')
    #for i in range(5):
        #fw1.write(str(TestACC[i]))
        #fw1.write(',')
    
    #fw1.close()

    fw2=open('OSmodel/result/OS3.1/test_result_model_3Glayerdim32early=5lr=1e-3epoch1000fc1.txt','w')
    for i in range(5):
        fw2.write(str(TestnewAUC[i]))
        fw2.write(',')
    fw2.write('\n')
    for i in range(5):
        fw2.write(str(TestnewAUPR[i]))
        fw2.write(',')
    fw2.write('\n')
    for i in range(5):
        fw2.write(str(TestnewACC[i]))
        fw2.write(',')
    
    fw2.close()
        


    
      