#Compute perplexity for every predicted encoded sequences based on model output
import numpy as np
def compute_perplexity(encoded_method_name,probability_distribution,bos_token=0,eos_token=2):
    log_sum = 0
    sentence_len = len(encoded_method_name)

    for idx,token in enumerate(encoded_method_name):
        if token == bos_token:
            continue
        log_sum += np.log(probability_distribution[idx][token])
        if token == eos_token:
            break
    
    return np.exp(-log_sum/sentence_len)