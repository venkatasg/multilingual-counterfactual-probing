"""
Running AlterRep on a model.
"""

import argparse
import logging
import os
from accelerate import Accelerator
import torch
import datasets
import numpy as np
from datasets import logging as dlogging
import ipdb
from inlp.debias import debias_by_specific_directions
from mlama_reader import MLama
from tqdm import tqdm
import re
import pandas as pd

import transformers
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    set_seed
)



dlogging.disable_progress_bar()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract representations from a model finetuned on a text classification task (POS) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
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
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="bert-base-multilingual-cased or xlm-roberta-base",
        default='bert-base-multilingual-cased'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
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
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--num_classifiers", 
        type=int, 
        default=32, 
        help="Number of iNLP iterations at which to slice."
    )
    parser.add_argument(
        "--total_classifiers", 
        type=int, 
        default=50, 
        help="Total Number of iNLP iterations."
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        help="Alpha value for intervention. Set only positive values, this script does both positive and negative interventions.", 
        required=True
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
        '--control', 
        action='store_true',
        help='Experiment with control intervention rather than actual intervention'
    )
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Experiment with test set rather than validation set'
    )
    parser.add_argument(
        '--random_words', 
        action='store_true', 
        default=False, 
        help='Carry out true random word substituion'
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="mixed/reps_inlp/",
        help="path containing inlp weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mixed/outputs/",
        help="folder to output results to"
    )
    parser.add_argument(
        "--mlama_path",
        type=str,
        default="data/mlama1.1/",
        help="folder containing mlama data"
    )
    parser.add_argument(
        '--random_words_path', 
        type=str, 
        default='data/muse/', 
        help='Folder containing words list'
    )

    args = parser.parse_args()
    
    return args

def get_probabilities(input_string, model, tokenizer):
    with torch.no_grad():
        output = model(**tokenizer(input_string, return_tensors='pt').to(model.device))
    log_probs = torch.log_softmax(output[0], dim=2)[:, 1:-1]
    return log_probs


def get_target_probabilities(template, targets, model, tokenizer):
    # tokenized_template = tokenizer.tokenize(template)
    tokenized_template = tokenizer.convert_ids_to_tokens(tokenizer.encode(template, add_special_tokens=False))
    mask_token = tokenizer.mask_token
    mask_index = tokenized_template.index(mask_token)
    ret = []
    for target in targets:
        tokenized = tokenizer.tokenize(target)
        num_tokens = len(tokenized)
        modded_template = re.sub(re.escape(mask_token), mask_token * num_tokens, template)
        assert len(tokenizer.tokenize(modded_template)) == len(tokenizer.tokenize(template)) + num_tokens - 1
        log_probs = get_probabilities(modded_template, model, tokenizer)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        target_probs = 0
        for i in range(num_tokens):
            target_probs += log_probs[0, mask_index + i, ids[i]].item()
        target_probs = target_probs / num_tokens
        ret.append(target_probs)
    return ret

def intervention(h_out, P, ws, alpha):
    '''
    Perform amnesic, positive or negative intervention on all tokens
    alpha=0 : Amnesic
    alpha>0 : Positive
    alpha<0 : Negative
    '''

    h_alter = h_out[0][:,1:,:]
        
    signs = torch.sign(h_alter@ws.T).long()
    
    # h_r component
    proj = (h_alter@ws.T) 
    if alpha>=0:
        proj = proj * signs
    else:
        proj = proj * (-signs)
    h_r = (proj@ws)*np.abs(alpha)
    
    # Get vector only in the direction of perpendicular to decision boundary
    h_n = h_alter@P

    # Now pushing it either in positive or negative intervention direction
    h_alter = h_n + h_r
        
    h_final = torch.cat((h_out[0][:,:1,:], h_alter), dim=1)
   
    return (h_final,)

def main():

    # Parse all args
    args = parse_args()
    
    lang_1 = args.langs.split('_')[0]
    lang_2 = args.langs.split('_')[1]
    
    # Set random seed
    set_seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Loading MLama
    ml = MLama(args.mlama_path)
    ml.load()
    ml.fill_all_templates('x')

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model_nohook = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path
        )
    model_pos = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path
        )
    model_neg = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path
        )
    
    # Extract mask token
    mask_token = tokenizer.mask_token
        
    # Use the device given by the `accelerator` object.
    device = accelerator.device

    # Prepare everything with our `accelerator`.
    model_nohook = accelerator.prepare(model_nohook)
    model_pos = accelerator.prepare(model_pos)
    model_neg = accelerator.prepare(model_neg)
    
    input_path = args.input_path + args.model_name_or_path + '/'
        
        
    # Load iNLP parameters
    if not args.control:
        with open(input_path + "Ws.langs={}.model={}.total_iters={}.npy".format(args.langs, args.model_name_or_path, args.total_classifiers), "rb") as f:
            Ws = np.load(f)
    else:
        with open(input_path + "Ws.langs={}.model={}.total_iters={}.npy".format(args.langs, args.model_name_or_path, args.total_classifiers), "rb") as f:
            Ws = np.load(f)

    # Reduce Ws to number of classifiers you want to set it to
    Ws = Ws[:args.num_classifiers,:]

    # Now derive P from Ws
    list_of_ws = [np.array([Ws[i,:]]) for i in range(Ws.shape[0])]
    P = debias_by_specific_directions(directions=list_of_ws, input_dim=Ws.shape[1])

    Ws = torch.tensor(Ws/np.linalg.norm(Ws, keepdims = True, axis = 1)).to(torch.float32).squeeze().to(device)
    P = torch.tensor(P).to(torch.float32).to(device)

    # Insert newaxis for 1 classifier edge case
    if len(Ws.shape)==1:
        Ws = Ws[np.newaxis,:]
    
    if args.model_name_or_path=='bert-base-multilingual-cased':
        hook_pos = model_pos.bert.encoder.layer[args.layer].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=args.alpha))
        hook_neg = model_neg.bert.encoder.layer[args.layer].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=-args.alpha))
    else:
        hook_pos = model_pos.roberta.encoder.layer[args.layer].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=args.alpha))
        hook_neg = model_neg.roberta.encoder.layer[args.layer].register_forward_hook(lambda m, h_in, h_out: intervention(h_out=h_out, P=P, ws=Ws, alpha=-args.alpha))
    model_nohook.eval()
    model_pos.eval()
    model_neg.eval()
    
    template_categories = ['P17', 'P19', 'P20', 'P27', 'P30', 'P36', 'P47', 'P131', 'P138', 'P159', 'P190', 'P276', 'P495', 'P530', 'P740', 'P937', 'P1001', 'P1376', 'P37', 'P103', 'P364', 'P407', 'P1412', 'P101', "P106", 'P140', 'P108', 'P127', 'P176', 'P178', 'P39', 'P136', 'P264', 'P1303', 'P31', 'P279', 'P361', 'P463', 'P527', 'P413', 'P449']
    
    words = {}
    for lang in ['en', 'hi', 'ko']:
        words_d = {}
        if args.random_words:
            words_l = np.loadtxt(args.random_words_path + '{}-filtered.txt'.format(lang), dtype=str).tolist()
        else:
            words_l = []
            for relation in tqdm(template_categories):
                for key in ml.data[lang][relation]['triples'].keys():
                    word = ml.data[lang][relation]['triples'][key]['obj_label']
                    words_l.append(word)
        for word in words_l:
            length = len(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
            if length in words_d:
                words_d[length].add(word)
            else:
                words_d[length] = set([word])
        
        for length in words_d:
            words_d[length] = list(words_d[length])
        words[lang] = words_d
    
    logprobs = [[[[] for ___ in range(3)] for __ in range(4)] for _ in range(2)]
    headers = ["langs", "model", "num_classifiers",  "random_word", "template", "answer", "push_to", "alpha", "logprob_baseline", "logprob_after_intervention"]
    out_data = []
    
    for relation in template_categories:
        pbar = (ml.data[lang_1][relation]['triples'].keys() & ml.data[lang_2][relation]['triples'].keys())
        for i, key in enumerate(pbar):
            
            lang_1_template = ml.data[lang_1][relation]['filled_templates'][key]
            lang_2_template = ml.data[lang_2][relation]['filled_templates'][key]
            lang_1_template = lang_1_template.replace('[Y]', mask_token)
            lang_2_template = lang_2_template.replace('[Y]', mask_token)

            lang_1_answer = ml.data[lang_1][relation]['triples'][key]['obj_label']
            # Set random answer to same number of tokens as correct answer
            lang_1_answer_len = len(tokenizer.convert_ids_to_tokens(tokenizer.encode(lang_1_answer, add_special_tokens=False)))
            # Following code ensures random answer is different from correct answer
            if lang_1_answer_len not in words[lang_1] or len(words[lang_1][lang_1_answer_len]) == 1:
                continue
            else:
                while True:
                    lang_1_random_answer = np.random.choice(words[lang_1][lang_1_answer_len])
                    if lang_1_random_answer != lang_1_answer:
                        break
    
            lang_2_answer = ml.data[lang_2][relation]['triples'][key]['obj_label']
            # Set random answer to same number of tokens as correct answer
            lang_2_answer_len = len(tokenizer.convert_ids_to_tokens(tokenizer.encode(lang_2_answer, add_special_tokens=False)))
            if lang_2_answer_len not in words[lang_2] or len(words[lang_2][lang_2_answer_len]) == 1:
                continue
            else:
                while True:
                    lang_2_random_answer = np.random.choice(words[lang_2][lang_2_answer_len])
                    if lang_2_random_answer != lang_2_answer:
                        break
                        
            
            for i, (lang, template) in enumerate(((lang_1, lang_1_template), (lang_2, lang_2_template))):
                if lang == lang_1:
                    itr = (lang_1, 0, lang_1_answer), (lang_1, 1, lang_1_random_answer), (lang_2, 0, lang_2_answer), (lang_2, 1, lang_2_random_answer)
                else:
                    itr = (lang_2, 0, lang_2_answer), (lang_2, 1, lang_2_random_answer), (lang_1, 0, lang_1_answer), (lang_1, 1, lang_1_random_answer)
                for j, (lang_answer, is_random, answer) in enumerate(itr):
                    probs_array = []
                    for k, model in enumerate((model_nohook, model_pos, model_neg)):
                        prob = get_target_probabilities(template, [answer], model, tokenizer)[0]
                        probs_array.append(prob)
                        # logprobs[i][j][k].append(prob)
                        
                    out_data.append([args.langs, args.model_name_or_path, args.num_classifiers, is_random, lang, lang_answer, lang_2, args.alpha, probs_array[0], probs_array[1]])
                    out_data.append([args.langs, args.model_name_or_path, args.num_classifiers, is_random, lang, lang_answer, lang_1, args.alpha, probs_array[0], probs_array[2]])
    
    output_path = args.output_path + '/'
    
    pd.DataFrame(data=out_data, columns=headers).to_csv(output_path + 'output_langs=' + args.langs + '_model=' + args.model_name_or_path + '_alpha=' + str(args.alpha) + '_num_classifiers=' + str(args.num_classifiers) + '_rand=' + str(args.random_words) + '_control=' + str(args.control) + '.csv', index=False)

    hook_pos.remove()
    hook_neg.remove()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
