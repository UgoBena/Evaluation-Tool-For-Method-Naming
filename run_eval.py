#Entry point to be run by the user to evaluate his predictions

import numpy as np
import pandas as pd

from semantic_analysis import levenshtein_distance,compute_cosine_similarity
from f1_values import get_acc_and_f1_values
from perplexity import compute_perplexity
from json_parser import JSONParser

import argparse
import json

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def main(args):
    do_printing = args["printing"]
    do_perplexity_computation = args["perplexity"]
    output_file = args["output_file"]
    

    #data to be output in the csv
    all_labels = []
    all_pred = []
    all_matches = []
    all_f1 = []
    all_levenshtein = []
    all_bert_dist = []
    all_perplexity = []

    #############################
    ### Parse json data
    #############################
    print(color.BOLD + "#"*20)
    print( "Parsing data" )
    print("#"*20 + color.END)
    print()

    predictions_list, encoded_predictions_list, softmax_outputs_list,label_list = JSONParser.get_lists(args["json_file"],do_perplexity_computation)

    #############################
    ### Compute evaluations
    #############################
    print(color.BOLD + "#"*20)
    print( "Evaluation Start" )
    print("#"*20 + color.END)
    print()

    for i in range(len(predictions_list)):
        label = label_list[i]
        print(color.UNDERLINE + f"Evaluation n°{i+1}" + color.END)
        print(f"Label : {label}")

        predictions = predictions_list[i]
        if do_perplexity_computation:
            encoded_predictions = encoded_predictions_list[i]
            softmax_outputs = softmax_outputs_list[i]

        max_f1_score = 0
        max_f1_score_idx = 0

        max_precision = 0
        max_precision_idx = 0

        max_recall = 0
        max_recall_idx = 0


        max_bert_similarity = np.inf
        max_bert_similarity_idx = 0

        min_levenshtein_distance = np.inf
        min_levenshtein_distance_idx = 0
        
        min_perplexity = np.inf
        min_perplexity_idx = 0

        exact_match_idx = None
        for j in range(len(predictions)):
            prediction = predictions[j]
            if do_printing:
                print("_"*20)
                print()
                print(color.UNDERLINE + f"Prediction n°{j+1}" + color.END)
                print(f"Predicted method name : {prediction}")
            if do_perplexity_computation:
                encoded_prediction = encoded_predictions[j]
                softmax_output = softmax_outputs[j]
            
            levenshtein_distance_value = levenshtein_distance(prediction,label)
            bert_similarity = compute_cosine_similarity(prediction,label)
            
            acc,f1,precision,recall = get_acc_and_f1_values(prediction,label)
            if do_perplexity_computation:
                perplexity = get_perplexity(prediction,softmax_output)
            

            #round values
            f1 = round(f1,2)
            precision = round(precision,2)
            recall = round(recall,2)
            levenshtein_distance_value = round(levenshtein_distance_value,2)
            bert_similarity = round(bert_similarity,2)
            if do_perplexity_computation:
                perplexity = round(perplexity,2)
            if (do_printing):
                print(f"Exact Match: {acc==1.}")
                print(f"F1 score: {f1}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                if do_perplexity_computation:
                    print(f"Perplexity: {perplexity}")
                print(f"Levenshtein Distance: {levenshtein_distance_value}")
                print(f"Cosine similarity between Bert vectors: {bert_similarity}")
                print("_"*20)
            

            if acc==1.:
                exact_match_idx = j + 1
            if f1 > max_f1_score:
                max_f1_score = f1
                max_f1_score_idx = j + 1
            if precision > max_precision:
                max_precision = precision
                max_precision_idx = j + 1
            if recall > max_recall:
                max_recall = recall
                max_recall_idx = j + 1
            if bert_similarity > max_bert_similarity:
                max_bert_similarity = bert_similarity
                max_bert_similarity_idx = j + 1
            if levenshtein_distance_value < min_levenshtein_distance:
                min_levenshtein_distance = levenshtein_distance_value
                min_levenshtein_distance_idx = j + 1
            if do_perplexity_computation and perplexity < min_perplexity:
                min_perplexity = perplexity
                min_perplexity_idx = j + 1
            
            #append to list for csv formatting
            all_labels.append(label)
            all_pred.append(prediction)
            all_matches.append(acc==1.)
            all_f1.append(f1)
            all_levenshtein.append(levenshtein_distance_value)
            all_bert_dist.append(bert_similarity)
            if do_perplexity_computation:
                all_perplexity.append(perplexity)

        if (do_printing):
            print(color.BOLD + "_"*20)
            print()
            print( "Global evaluation metrics")
            print("_"*20 + color.END)
            if exact_match_idx is not None:
                print(f"Exact Match found for prediction n° {exact_match_idx}")
            else:
                print(f"Top-{len(predictions)} Best F1 score: {round(max_f1_score,2)} achieved for prediction n° {max_f1_score_idx}")
                print(f"Top-{len(predictions)} Best Recall: {round(max_recall,2)} achieved for prediction n° {max_recall_idx}")
                print(f"Top-{len(predictions)} Best Precision: {round(max_precision,2)} achieved for prediction n° {max_precision_idx}")
                print(f"Top-{len(predictions)} Minimal Levenshtein Distance: {round(min_levenshtein_distance,2)} achieved for prediction n° {min_levenshtein_distance_idx}")
                print(f"Top-{len(predictions)} Max Cosine similarity between Bert vectors: {round(max_bert_similarity,2)} achieved for prediction n° {max_bert_similarity_idx}")
            if do_perplexity_computation:
                print(f"Top-{len(predictions)} Minimal Perplexity: {round(min_perplexity,2)} achieved for prediction n° {min_perplexity_idx}")
            print("#"*20)
    df = pd.DataFrame({
        "Label":all_labels,
        "Prediction":all_pred,
        "Exact Match": all_matches,
        "F1 Score": all_f1,
        "Levenshtein distance":all_levenshtein,
        "Bert cosine similarity":all_bert_dist
    })
    df.to_csv(output_file,index=False)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    #############################
    ### Parse arguments
    #############################
    parser = argparse.ArgumentParser(description='Evaluate method name prediction contained in a json (see Readme for details on the json formatting)')
    parser.add_argument('-i','--json_file', type=str,
                        help='the path to the input json file')
    parser.add_argument('--no-perplexity', dest='perplexity', action='store_const',
                        const=False, default=True,
                        help='Do not compute perplexity')
    parser.add_argument("-o", '--output_file', type=str, default="output.csv",
                        help='output csv file to store results in')
    parser.add_argument("-p",'--add_print',dest='printing', action='store_const',
                        const=True, default=False,
                        help='Print the results in the standard output')
    args = parser.parse_args()
    arg_dict = vars(args)
    print("Script started with following args:")
    print(arg_dict)
    #make sure the path extension is .json
    assert arg_dict["json_file"].split(".")[-1] == "json"
    assert arg_dict["output_file"].split(".")[-1] == "csv"

    main(arg_dict)