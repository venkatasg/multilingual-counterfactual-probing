"""
Extract representations from a model with accelerate library.
"""

import argparse
import random
import numpy as np
import ipdb
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, logging as dlogging
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    default_data_collator,
    set_seed,
    logging
)

dlogging.disable_progress_bar()
logging.set_verbosity(40)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract representations from a model finetuned on a text classification task (POS) with accelerate library"
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed."),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-multilingual-cased',
        help="bert-base-multilingual-cased or xlm-roberta-base",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--layer", 
        type=int, 
        default=11, 
        help="Layer from which to extract hidden states."
    )
    parser.add_argument(
        "--langs", 
        type=str, 
        required=True, 
        help="language pairs on which to train language classifier."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to text from which to extract token representations"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="folder to save representations to"
    )


    args = parser.parse_args()

    return args

def parse_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n\n')
        examples = []
        for example in lines:
            example = example.split('\n')
            tokens = [line.split('\t')[0] for line in example]
            labels = [line.split('\t')[-1] for line in example]
            examples.append({'tokens': tokens, 'labels': labels})
    examples = pd.DataFrame(examples)
    return examples

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
   
    # If passed along, set the training seed now.
    set_seed(args.seed)
    
    data_path = args.data_path + args.langs + '/'
        
    # First find unique labels and set up label2id
    label_list = [x.upper() for x in args.langs.split('_')]
    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    
    train_df = parse_data(data_path + 'train.txt')
    val_df = parse_data(data_path + 'val.txt')
    test_df = parse_data(data_path + 'test.txt')
    
    train_dataset_raw = Dataset.from_pandas(train_df)
    val_dataset_raw = Dataset.from_pandas(val_df)
    test_dataset_raw = Dataset.from_pandas(test_df)
    
    # Columns we will be using
    text_column_name = "tokens"
    label_column_name = "labels"

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            config=config
        )


    model.resize_token_embeddings(len(tokenizer))

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}
    
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
                
        mask_token_id = tokenizer.mask_token_id
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            
            # Replaced 15% of tokenized inputs with [MASK] token
            input_ids = tokenized_inputs.input_ids[i]
            for tok_ind in range(1,len(input_ids)-1):
                if random.random() < 0.15:
                    input_ids[tok_ind] = mask_token_id
            tokenized_inputs.input_ids[i] = input_ids
            
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label2id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    raw_datasets = DatasetDict({'train': train_dataset_raw, 'val': val_dataset_raw, 'test': test_dataset_raw})
    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_raw_datasets['train']
    val_dataset = processed_raw_datasets['val']
    test_dataset = processed_raw_datasets['test']

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size)

    # Use the device given by the `accelerator` object.
    device = accelerator.device

    # Prepare everything with our `accelerator`.
    model, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)

    model.eval()
    
    def get_reps(dataloader, label2id):
        
        reps_dict = {l: None for l in label2id.keys()}
        
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
    
            hidden_states = outputs["hidden_states"]
    
            labels = batch["labels"].cpu().detach().numpy()
            layer_states = hidden_states[args.layer].cpu().detach().numpy()
            
            for lang in label2id.keys():
                lang_id = label2id[lang]
                idg = (labels==lang_id).nonzero()
                lang_reps = None
                for ij in range(len(idg[0])):
                    if lang_reps is not None:
                        lang_reps = np.concatenate((lang_reps, layer_states[idg[0][ij], idg[1][ij], :][np.newaxis,:]), 0)
                    else:
                        lang_reps = layer_states[idg[0][ij], idg[1][ij], :][np.newaxis,:]
                if lang_reps is not None:
                    if reps_dict[lang] is not None:
                        reps_dict[lang] = np.concatenate((reps_dict[lang], lang_reps), 0)
                    else:
                        reps_dict[lang] = lang_reps
        
        return reps_dict

    train_reps_dict = get_reps(train_dataloader, label2id)
    val_reps_dict = get_reps(val_dataloader, label2id)
    test_reps_dict = get_reps(test_dataloader, label2id)
    
    output_path = args.output_path + args.langs + '/' + args.model_name_or_path + '/'
    for lang in label2id.keys():
        with open(output_path + lang + '_val.npy', 'wb') as f:    
            np.save(f, val_reps_dict[lang])
            
        with open(output_path + lang + '_test.npy', 'wb') as f:
            np.save(f, test_reps_dict[lang])

        with open(output_path + lang + '_train.npy', 'wb') as f:
            np.save(f, train_reps_dict[lang])
    

if __name__ == "__main__":
    main()
