#Script to compute the levenshtein distance between two method names
from nltk.corpus import wordnet
import numpy as np
import re

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

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def levenshtein_distance(prediction,label):
    #First turn the method strings into lists
    subtokenized_prediction = subTokenize(prediction)
    subtokenized_label = subTokenize(label)

    #Run recursive algorithm to find the distance
    distance = levenshtein_rec(subtokenized_prediction,subtokenized_label)

    return distance

def levenshtein_rec(list_a,list_b,insertion_weight=.5,deletion_weight=.5,synonyms_weight=.5):
    if len(list_a) == 0:
        #those are insertions in a that weights .5
        return insertion_weight*len(list_b)
    if len(list_b) == 0:
        #those are deletions from a that weight .5
        return deletion_weight*len(list_a)
    if list_a[0] == list_b[0]:
        return levenshtein_rec(list_a[1:],list_b[1:])
    
    #Compute the distance in each 3 possible case (1: deletion from a, 2: insertion in a, 3: replacement)
    results = [levenshtein_rec(list_a[1:],list_b),
            levenshtein_rec(list_a,list_b[1:]),
            levenshtein_rec(list_a[1:],list_b[1:])]
    action = np.argmin(results)
    #case deletion
    if action == 0:
        return deletion_weight + results[action]
    #case insertion
    if action == 1:
        return insertion_weight + results[action]
    #case replacement, first check if the words are synonyms
    if action == 2:
        if list_a[0] in get_synonyms(list_b[0]):
            return synonyms_weight + results[action]
        else:
            return 1 + results[action]
