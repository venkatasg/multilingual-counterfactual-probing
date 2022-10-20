# multilingual-counterfactual-probing
Code related to multilingual counterfactual probing

# Steps to reproduce experiment results

Given a language pair X-Y, we use the following pipeline to train an iNLP classifier which can then perform the intervention to change language information:

1. Collect hidden state representations of tokens on data and for a specific model, with their corresponding language ID labels.
2. Train an iNLP classifier on the saved tokens and labels.
3. Perform the AlterRep operation on test data and store predictions.

## Storing token representations

`mult_collect_states.py` collects hidden state representations from a transformer based language model for the two language pairs and saves them to a folder. By default the 11th layer of a `bert-base-multilingual-cased` model are extracted, but these can be changed with command line arguments.

```
python mult_collect_states.py --langs LANG_PAIR --data_path DATA_PATH --output_path OUTPUT_PATH
```

## Training iNLP

`mult_inlp` trains iNLP classifiers using the tokens and labels saved from the previous step. To specify the number of iterations you wish to train iNLP for, pass it to the `num_classifiers` argument when calling the script.

```
python mult_inlp.py --data_path DATA_PATH --langs LANG_PAIR --num_classifiers NUM_CLASSIFIERS --output_path OUTPUT_PATH
```

## Create test data

Use the script `mult_mono_create_test_data.py` to create the test data for the non-code mixed case that was used in the paper. It saves the created test files in `DATA_PATH`

```
python mult_mono_create_test_data.py --langs --LANG_PAIR --data_path DATA_PATH
```

## Running AlterRep intervention

`mult_alter.py` and `mult_alter_mixed.py` generate CSV output files in the [long data]() format -- each row is a datapoint from the test file, and each column is a variable.

```
python mult_alter.py --langs LANG_PAIR --data_path DATA_PATH --num_classifiers NUM_CLASSIFIERS --alpha 4 --output_path OUTPUT_PATH
```


To aggregate all the output CSVs from different models, alphas, random/control interventions, and number of classifiers, use the `aggregate_data.py` script:

```
python aggregate_data.py --output_dir mono_outputs
```