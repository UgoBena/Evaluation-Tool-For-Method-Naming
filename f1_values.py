#Compute accuracy, f1 score, at TOP1 and TOP5 
import numpy as np

def get_acc_and_f1_values(encoded_pred,encoded_tgt):
    #metrics to compute
    precision = 0
    recall = 0
    acc = 0

    pred = np.array(encoded_pred)
    tgt = np.array(encoded_tgt)
    
    tp = float((np.isin(pred,tgt)*1).sum())
    fp = float((np.isin(pred,tgt,invert=True)*1).sum())
    fn = float((np.isin(tgt,pred,invert=True)*1).sum())

    #Precision
    if (tp + fp != 0.): precision += tp/(tp + fp)
    #Recall
    if (tp + fn != 0.): recall += tp/(tp + fn)
    #Acc
    acc += (fp==0. and fn==0.) * 1.

    if precision + recall != 0.:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.

    return acc,f1,precision,recall