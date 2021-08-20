#Compute accuracy, f1 score, at TOP1 and TOP5 
import numpy as np
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

def tokenize_and_remove_underscores(sequence):
    underscore_token = tokenizer.get_vocab()["_"]
    encoded_with_underscore = tokenizer.encode_plus(sequence,add_special_tokens=False)["input_ids"]
    encoded = [token for token in encoded_with_underscore if token != underscore_token]
    return encoded



def get_acc_and_f1_values(prediction,label):
    encoded_pred = tokenize_and_remove_underscores(prediction)
    encoded_tgt = tokenize_and_remove_underscores(label)
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