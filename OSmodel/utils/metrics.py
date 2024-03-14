import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,auc,precision_recall_curve,f1_score,matthews_corrcoef,accuracy_score
import copy

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def AUPR(ground_truth,prediction):
    precision,recall,thresholds=precision_recall_curve(ground_truth, prediction)
    res=auc(recall,precision)
    #res = average_precision_score(y_true=ground_truth, y_score=prediction)
    return(res)

def F1(ground_truth, prediction):
    #if prediction + ground_truth > 0:
        #return (2.0 * prediction * ground_truth) / (prediction + ground_truth)
    #else:
        #return 0.
        
    return f1_score(ground_truth,[int(i>0.5) for i in prediction ])

def MCC(ground_truth,prediction):
    return matthews_corrcoef(ground_truth,[int(i>0.5) for i in prediction ])

def ACC(ground_truth,prediction):
    
    fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    # 利用Youden's index计算阈值
    spc = 1 - fpr
    j_scores = tpr - fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    
    youden_thresh = round(youden_thresh, 3)
    return accuracy_score(ground_truth,[int(i>=youden_thresh) for i in prediction ])