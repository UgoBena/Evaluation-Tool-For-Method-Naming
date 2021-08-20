#Entry point to be run by the user to evaluate his predictions

from transformers import RobertaTokenizer
import torch
import numpy as np

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
    predictions_list, encoded_predictions_list, softmax_outputs_list,label_list = JSONParser.get_lists(args["json_file"],do_perplexity_computation)

    #############################
    ### Compute evaluations
    #############################
    codeBERTa_embeddings = torch.load("./codeBERTa_embeddings.pt")
    codeBERTa_tokenizer = RobertaTokenizer.from_pretrained("./codeBERTa_tokenizer")

    print(color.BOLD + "#"*20)
    print( "Evaluation Start" )
    print("#"*20 + color.END)
    print()

    for i in range(len(predictions_list)):
        label = label_list[i]
        print(color.UNDERLINE + f"Evaluation n°{i+1}" + color.END)
        print(f"Label : {label}")

        predictions = predictions_list[i]
        encoded_predictions = encoded_predictions_list[i]
        softmax_outputs = softmax_outputs_list[i]

        encoded_label = tokenize_and_remove_underscores(label,codeBERTa_tokenizer)
        max_f1_score = 0
        max_f1_score_idx = 0

        max_precision = 0
        max_precision_idx = 0

        max_recall = 0
        max_recall_idx = 0


        min_distance_between_average = np.inf
        min_distance_between_average_idx = 0

        min_average_min_distance = np.inf
        min_average_min_distance_idx = 0
        
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
            encoded_prediction = encoded_predictions[j]
            softmax_output = softmax_outputs[j]
            
            codeBERTa_encoded_pred = tokenize_and_remove_underscores(prediction,codeBERTa_tokenizer)
            
            average_min_distance,distance_between_average= semantic_proximity_codeBERTa(codeBERTa_encoded_pred,encoded_label,codeBERTa_tokenizer,codeBERTa_embeddings)
            acc,f1,precision,recall = get_acc_and_f1_values(codeBERTa_encoded_pred,encoded_label)
            perplexity = get_perplexity(encoded_prediction,softmax_output)

            #round values
            f1 = round(f1,2)
            precision = round(precision,2)
            recall = round(recall,2)
            average_min_distance = round(average_min_distance,2)
            distance_between_average = round(distance_between_average,2)
            perplexity = round(perplexity,2)
            if (do_printing):
                print(f"Exact Match: {acc==1.}")
                print(f"F1 score: {f1}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"Perplexity: {perplexity}")
                print(f"Levenshtein Distance: {average_min_distance}")
                print(f"Distance between BERT sequence vectors: {distance_between_average}")
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
            if average_min_distance < min_average_min_distance:
                min_average_min_distance = average_min_distance
                min_average_min_distance_idx = j + 1
            if distance_between_average < min_distance_between_average:
                min_distance_between_average = distance_between_average
                min_distance_between_average_idx = j + 1
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                min_perplexity_idx = j + 1

        if (do_printing):
            print(color.BOLD + "_"*20)
            print()
            print( "Global evaluation metrics")
            print("_"*20 + color.END)
            if exact_match_idx is not None:
                print(f"Exact Match found for prediction n° {exact_match_idx}")
            else:
                print(f"Top-{len(predictions)} F1 score: {round(max_f1_score,2)} achieved for prediction n° {max_f1_score_idx}")
                print(f"Top-{len(predictions)} Recall: {round(max_recall,2)} achieved for prediction n° {max_recall_idx}")
                print(f"Top-{len(predictions)} Precision: {round(max_precision,2)} achieved for prediction n° {max_precision_idx}")
                print(f"Top-{len(predictions)} Minimal Levenshtein Distance: {round(min_average_min_distance,2)} achieved for prediction n° {min_average_min_distance_idx}")
                print(f"Top-{len(predictions)} Minimal Distance between BERT sequence vectors: {round(min_distance_between_average,2)} achieved for prediction n° {min_distance_between_average_idx}")

            print(f"Top-{len(predictions)} Minimal Perplexity: {round(min_perplexity,2)} achieved for prediction n° {min_perplexity_idx}")
            print("#"*20)


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
                        const=False, default=True,
                        help='Print the results in the standard output')
    args = parser.parse_args()
    arg_dict = vars(args)
    #make sure the path extension is .json
    assert arg_dict["json_file"].split(".")[-1] == "json"
    assert arg_dict["output_file"].split(".")[-1] == "csv"

    main(arg_dict)