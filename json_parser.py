import json

class JSONParser():
    @classmethod
    def get_lists(cls,json_file,do_perplexity_computation=True):
        #############################
        ### Parse JSON file
        #############################
        with open(json_file) as f:
            data = json.load(f)

        predictions_list = []
        label_list = []
        if do_perplexity_computation:
            encoded_predictions_list = []
            softmax_outputs_list = []
        
        for i,evaluation in enumerate(data):
            label_list.append(evaluation["label"])

            predictions_list.append([])
            if do_perplexity_computation:
                encoded_predictions_list.append([])
                softmax_outputs_list.append([])
            for pred in evaluation["predictions"]:
                predictions_list[i].append(pred["prediction"])
                if do_perplexity_computation:
                    encoded_predictions_list[i].append(pred["encoded_prediction"])
                    softmax_outputs_list[i].append(torch.load(pred["softmax_output"],map_location=torch.device('cpu')))
        if not do_perplexity_computation:
            encoded_predictions_list= None
            softmax_outputs_list = None
        return predictions_list, encoded_predictions_list, softmax_outputs_list,label_list