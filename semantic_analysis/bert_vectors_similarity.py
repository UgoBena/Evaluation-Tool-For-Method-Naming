#Script to compute the bert vector distance between two method names
from transformers import AutoTokenizer, AutoModel
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def subTokenize(function_name):
    subtokens = function_name.split("_")
    if subtokens[0] == '': subtokens.pop(0)
    #Check if name contained underscore (snake_case)
    #Otherwise assume it's camelCase or PascalCase
    if len(subtokens) == 1:
        subtokens = re.findall('^[a-z]+|[A-Z][^A-Z]*',function_name)
    for i in range(len(subtokens)):
        subtokens[i] = subtokens[i].lower()
    return subtokens

def get_tokens(prediction,label):
    #the chosen model takes as input seqences of size 128
    input_seq_len = 128
    #put space between each subtokens
    prediction = " ".join(subTokenize(prediction))
    label = " ".join(subTokenize(label))

    tokens = {'input_ids': torch.zeros(2,input_seq_len).long(), 'attention_mask': torch.zeros(2,input_seq_len).long()}

    prediction_tokens = tokenizer.encode_plus(prediction, max_length=input_seq_len, truncation=True,
                                       padding='max_length', return_tensors='pt')
    
    label_tokens = tokenizer.encode_plus(label, max_length=input_seq_len, truncation=True,
                                       padding='max_length', return_tensors='pt')
    
    tokens['input_ids'][0] = prediction_tokens['input_ids'][0]
    tokens['input_ids'][1] = label_tokens['input_ids'][0]

    tokens['attention_mask'][0] = prediction_tokens['attention_mask'][0]
    tokens['attention_mask'][1] = label_tokens['attention_mask'][0]

    return tokens


def get_sentence_vector(tokens):
    #embeddings are given by the last layer of the model
    embeddings = model(**tokens).last_hidden_state
    #we want to mean pool those embeddings, but fisrt we need to mask the embeddings corresponding to padding tokens
    attention_mask = tokens["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled.detach().numpy()


def compute_cosine_similarity(prediction,label):
    tokens = get_tokens(prediction,label)

    mean_pooled = get_sentence_vector(tokens)

    return cosine_similarity(mean_pooled,mean_pooled)[0][1]
