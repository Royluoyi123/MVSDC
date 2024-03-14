from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time
import math
import random
from sklearn.metrics import roc_auc_score,roc_curve
import sklearn.metrics as m
import copy

cores = multiprocessing.cpu_count() // 2

def sigmoid_function(z):
    fz=[]
    for num in z:
        fz.append(1/(1+math.exp(-num)))
    return fz




def result_test(model,all_test):
    test_score=[]
    drug1_list=[]
    drug2_list=[]
    cell_list=[]
    label_list=[]
    for drug1_id,cell_id,drug2_id,label in all_test:
        drug1_list.append(int(drug1_id))
        drug2_list.append(int(drug2_id))
        cell_list.append(int(cell_id))
        label_list.append(int(label))
    #for drug1_id,cell_id,drug2_id in testneg_cf:
        #drug1_list.append(int(drug1_id))
        #drug2_list.append(int(drug2_id))
        #cell_list.append(int(cell_id))

    test_score=model.generate(np.array(drug1_list),np.array(drug2_list),np.array(cell_list)).detach().cpu()

    #truthpos=[1 for x in range(0,len(test_cf))]
    #truthneg=[0 for x in range(0,len(testneg_cf))]
    #truthpos.extend(truthneg)
    
    test_score=sigmoid_function(test_score)
    #print(truthpos,test_pos_score)

    #auc = roc_auc_score(y_true=truthpos, y_score=test_score)
    #p, r, t = precision_recall_curve(y_true=truthpos, probas_pred=test_score)
    #aupr = m.auc(r, p)
    #fpr, tpr, threshold = roc_curve(truthpos,test_score)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    #predicted_label = copy.deepcopy(test_score)
    #youden_thresh = round(youden_thresh, 3)
    #print(youden_thresh)

    #predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]
    
    aupr=AUPR(ground_truth=label_list,prediction=test_score)
    auc=AUC(ground_truth=label_list, prediction=test_score)
    #f1=F1(ground_truth=truthpos, prediction=test_score)
    #mcc=MCC(ground_truth=truthpos, prediction=test_score)
    acc=ACC(ground_truth=label_list, prediction=test_score)
    #print(aupr,auc)
        
    return auc,aupr,acc,test_score

def result_train(model,all_train):
    test_score=[]
    drug1_list=[]
    drug2_list=[]
    cell_list=[]
    label_list=[]
    for drug1_id,cell_id,drug2_id,label in all_train:
        drug1_list.append(int(drug1_id))
        drug2_list.append(int(drug2_id))
        cell_list.append(int(cell_id))
        label_list.append(int(label))
    #for drug1_id,cell_id,drug2_id in trainneg_cf:
        #drug1_list.append(int(drug1_id))
        #drug2_list.append(int(drug2_id))
        #cell_list.append(int(cell_id))

    test_score=model.generate(np.array(drug1_list),np.array(drug2_list),np.array(cell_list)).detach().cpu()

    #truthpos=[1 for x in range(0,len(train_cf))]
    #truthneg=[0 for x in range(0,len(trainneg_cf))]
    #truthpos.extend(truthneg)
    
    test_score=sigmoid_function(test_score)
    #print(truthpos,test_pos_score)


    #auc = roc_auc_score(y_true=truthpos, y_score=test_score)
    #p, r, t = precision_recall_curve(y_true=truthpos, probas_pred=test_score)
    #aupr = m.auc(r, p)
    #fpr, tpr, threshold = roc_curve(truthpos,test_score)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    #predicted_label = copy.deepcopy(test_score)
    #youden_thresh = round(youden_thresh, 3)
    #print(youden_thresh)

    #predicted_label = [1 if i >= youden_thresh else 0 for i in predicted_label]

    aupr=AUPR(ground_truth=label_list,prediction=test_score)
    auc=AUC(ground_truth=label_list, prediction=test_score)
    #f1=F1(ground_truth=truthpos, prediction=test_score)
    #mcc=MCC(ground_truth=truthpos, prediction=test_score)
    acc=ACC(ground_truth=label_list, prediction=test_score)
    #print(aupr,auc)
        
    return auc,aupr,acc,test_score

def result_vaild(model,vaild_cf,vaildneg_cf):
    test_score=[]
    drug_list1=[]
    drug_list2=[]
    cell_list=[]
    for drug_id1,drug_id2,cell_id in vaild_cf:
        drug_list1.append(int(drug_id1))
        drug_list2.append(int(drug_id2))
        cell_list.append(int(cell_id))
    for drug_id1,drug_id2,cell_id in vaildneg_cf:
        drug_list1.append(int(drug_id1))
        drug_list2.append(int(drug_id2))
        cell_list.append(int(cell_id))

    test_score=model.generate(np.array(drug_list1),np.array(drug_list2),np.array(cell_list)).detach().cpu()

    truthpos=[1 for x in range(0,len(vaild_cf))]
    truthneg=[0 for x in range(0,len(vaildneg_cf))]
    truthpos.extend(truthneg)
    
    test_score=sigmoid_function(test_score)
    #print(truthpos,test_pos_score)
    aupr=AUPR(ground_truth=truthpos,prediction=test_score)
    auc=AUC(ground_truth=truthpos, prediction=test_score)
    #f1=F1(ground_truth=truthpos, prediction=test_score)
    #mcc=MCC(ground_truth=truthpos, prediction=test_score)
    acc=ACC(ground_truth=truthpos, prediction=test_score)
    #print(aupr,auc)
        
    return auc,aupr,acc



