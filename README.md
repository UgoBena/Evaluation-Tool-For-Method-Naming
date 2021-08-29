# Evaluation-Tool-For-Method-Naming

The tool can be downloaded from the GitHub page and once installed, can be run easily by following the instructions.

First, the tool requires specific Python packages to be run, which can be downloaded using either pip or conda:
\begin{itemize}
    \item Pytorch (for computations)
    \item NumPy (for computations)
    \item Pandas (for csv formatting)
    \item HuggingFace (for cosine similarity metric) 
\end{itemize}

To get an evaluation, the ` run_eval.py ` script must be run with the following as inputs: `-i` or `--json_file` followed by the path to the input json file (see next paragraph for additional information on the input), `-o` or `--output_csv` followed by a path to csv file that will be created and will store all the evaluation's output, and some additional options that can be checked by running `python run_eval.py -h`.

Regarding the input formatting, there is an example of such JSON file within the project: the `model_file.json` file, which has the structure discribed by Listing \ref{json-example}:

```json
{
        "label":"funtion_name",
        "predictions":[
            {
                "prediction":"my_pred1",
                "encoded_prediction":[0,8,5,2],
                "softmax_output":"output_0_0.pt"
            },
            {
                "prediction":"aSecondPredInCamelCase",
                "encoded_prediction":[0,3,4,2],
                "softmax_output":"output_0_1.pt"
            }
        ]
    } ,

    {
        "label":"secondFunctionName",
        "predictions":[
            {
                "prediction":"ASinglePredForAnotherLabelInPascalCase",
                "encoded_prediction":[0,2,4,6],
                "softmax_output":"output_1_0.pt"
            },
            {
                "prediction":"yet_another_pred",
                "encoded_prediction":[0,2,4,8],
                "softmax_output":"output_1_1.pt"
            }
        ]
    }
]
```

Note that the JSON is not a dictionary but a list of dictionary, each containing the information for a single evaluation: the label, and then all the suggestions of names output by a model. We also require the encoded prediction and the model's softmax output to compute the perplexity. This can be skipped if one is not interested in computing perplexity. One would need to add `--no-perplexity` when running the command ` python run_eval.py`.

Any number of predictions can be given for each label, but this number must be the same for all of the predictions. This is required to be able to compute the TopN version of each metric.

The output of the script can be printed to standard output if the `--do-printing` flag is set to True. In any case, the script will build a csv file containing, for each line, the label, the prediction, and the metrics computed for this pair. A second csv file (with the same name as the first one and `_macro` appended to it) will contain macro information like the average metrics or their TopN version, where N is the number of suggestions per label given in the json input.
