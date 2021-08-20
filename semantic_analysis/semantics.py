#Script to compute the distance between two method names

import torch

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

#############################
### codeBERTa Embeddings
#############################
def tokenize_and_remove_underscores(method_name,codeBERTa_tokenizer):
    #do not add special tokens, we do not want to compute distance on BOS and EOS tokens.
    tokenized_method_name = codeBERTa_tokenizer.encode(method_name,add_special_tokens=False)
    #remove tokens corresponding to underscores
    underscore_token = codeBERTa_tokenizer.get_vocab()["_"]
    
    clean_tokenized_method_name = [token for token in tokenized_method_name if token != underscore_token]
    return clean_tokenized_method_name

def get_word_embeddings_codeBERTa(method_name,codeBERTa_tokenizer,codeBERTa_embeddings):
    clean_tokenized_method_name = tokenize_and_remove_underscores(method_name,codeBERTa_tokenizer)

    return codeBERTa_embeddings[clean_tokenized_method_name]

#Returns index and value of the distance for the embedding in embeddings_list with minimal distance to word_embedding
def get_min_distance(word_embedding,embeddings_list):
    distances = torch.norm(embeddings_list - word_embedding.unsqueeze(0), dim=1)
    return distances.argmin().item(),distances.min().item()

def get_average_min_distance(embeddings_list_1,embeddings_list_2):
    num_embeddings = embeddings_list_1.size(0)
    total_distance = 0
    for i in range(num_embeddings):
        total_distance+= get_min_distance(embeddings_list_1[i],embeddings_list_2)[1]

    return total_distance


def semantic_proximity_codeBERTa(method_name_1,method_name_2,codeBERTa_tokenizer,codeBERTa_embeddings):

    embeddings_list_1 = get_word_embeddings_codeBERTa(method_name_1,codeBERTa_tokenizer,codeBERTa_embeddings)
    embeddings_list_2 = get_word_embeddings_codeBERTa(method_name_2,codeBERTa_tokenizer,codeBERTa_embeddings)

    average_embedding_1 = embeddings_list_1.mean(dim=0)
    average_embedding_2 = embeddings_list_2.mean(dim=0)


    average_distance = torch.norm(average_embedding_1 - average_embedding_2).item()

    return get_average_min_distance(embeddings_list_1,embeddings_list_2),average_distance