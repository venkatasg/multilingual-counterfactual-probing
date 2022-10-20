"""
Running AlterRep positive and negative on multilingual data
"""

import argparse
import os
import sys
from accelerate import Accelerator
import torch
import numpy as np
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import ipdb
from inlp.debias import debias_by_specific_directions
from tqdm import tqdm
import re
import pandas as pd
from collections import Counter
from random import sample, seed
import csv

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    set_seed,
    logging
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="R"
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
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int, 
        default=1, 
        help="A seed for reproducible training."
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
        "--input_path", 
        type=str, 
        required=True, 
        help="Directory where data files are stored."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory in which to store the output files."
    )
    parser.add_argument(
        '--random_inlp', 
        action='store_true', 
        default=False, 
        help='Use random iNLP directions'
    )
    parser.add_argument(
        "--inlp_weights_path",
        type=str,
        default="mono/reps_inlp/",
        help="location of saved inlp weights"
    )

    args = parser.parse_args()
    
    return args

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


def prepare_test_data(input_path, lang, tokenizer, batch_size):
    
    
    test_df = pd.read_csv(input_path + lang + '_' + tokenizer.name_or_path + '-val' + '.tsv', sep='\t', quoting=csv.QUOTE_NONE, na_filter=False) 

    # Encode sentences 
    data_lang = Dataset.from_pandas(test_df)
    data_lang_true = data_lang.map(lambda x: tokenizer(x['masked_sent_true'], truncation=True, padding="max_length", max_length=256), batched=True)

    true_toks = data_lang['true_answer']   
    true_answer_tokens = [tokenizer(tok, add_special_tokens=False)['input_ids'] for tok in true_toks]
    true_answer_tokens = [true_answer_tokens[i:i+batch_size] for i in range(0, len(true_answer_tokens), batch_size)]
    
    data_lang_true = data_lang_true.remove_columns(['masked_sent_true', 'masked_sent_transl', 'true_answer', 'transl_answer', 'true_control', 'transl_control', 'true_random', 'transl_random'])
    data_lang_true.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return data_lang_true, true_answer_tokens



def main():
    
    # Only print errors
    logging.set_verbosity(40)
    disable_progress_bar()
    
    # Parse all args
    args = parse_args()
    
    lang_1 = args.langs.split('_')[0]
    lang_2 = args.langs.split('_')[1]
    
    # Set random seed
    set_seed(args.seed)
    seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    # Use the device given by the `accelerator` object.
    device = accelerator.device
    
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
    
    # Prepare everything with our `accelerator`.
    model_nohook, model_pos, model_neg = accelerator.prepare(model_nohook,model_pos, model_neg)
    
    if args.num_classifiers > 0:
        # Load iNLP parameter Ws
        input_path = args.inlp_weights_path
        if not args.random_inlp:
            with open(input_path + "Ws.langs={}.model={}.total_iters={}.npy".format(args.langs, args.model_name_or_path, args.total_classifiers), "rb") as f:
                Ws = np.load(f)
        else:
            with open(input_path + "Ws.langs={}.model={}.total_iters={}.rand.npy".format(args.langs, args.model_name_or_path, args.total_classifiers), "rb") as f:
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
    
    ###### Make dataset #########
    data_lang_1_true, true_answer_tokens_1 = prepare_test_data(input_path=args.input_path, lang=lang_1, tokenizer=tokenizer, batch_size=args.batch_size)
    data_lang_2_true, true_answer_tokens_2 = prepare_test_data(input_path=args.input_path, lang=lang_2, tokenizer=tokenizer, batch_size=args.batch_size)
    
    dataloader_lang_1_true = torch.utils.data.DataLoader(data_lang_1_true, batch_size=args.batch_size, shuffle=False)
    dataloader_lang_2_true = torch.utils.data.DataLoader(data_lang_2_true, batch_size=args.batch_size, shuffle=False)
    
    dataloader_lang_1_true, dataloader_lang_2_true = accelerator.prepare(dataloader_lang_1_true, dataloader_lang_2_true)
    
    
    model_nohook.eval()
    model_pos.eval()
    model_neg.eval()
    
    final_df = {'lang': []}
    for intervention_type in ['baseline', 'pos', 'neg']:
        for top_k in ['_top_1', '_top_5', '_top_10', '_top_50', '_top_100']:
            final_df[intervention_type + top_k] = []
    lm_acc = {}

    mask_token_id = tokenizer.mask_token_id
    
    experiment_conditions = [
        (lang_1, dataloader_lang_1_true, true_answer_tokens_1), 
        (lang_2, dataloader_lang_2_true, true_answer_tokens_2)
    ]
    
    with torch.no_grad():
        for lang, dataloader, answer_tokens in experiment_conditions:
            for (intervention_type, model_test) in [('baseline', model_nohook), ('pos', model_pos), ('neg', model_neg)]:
                
                num_of_examples = 0
                for top_k in ['_top_1', '_top_5', '_top_10', '_top_50', '_top_100']:
                    lm_acc[top_k] = 0
                
                for batch_ind, input_dict in enumerate(dataloader):
                    
                    toks = answer_tokens[batch_ind]
                    
                    output = model_test(**input_dict)
                    log_probs = torch.log_softmax(output['logits'], dim=2)
                    
                    
                    for ind in range(log_probs.shape[0]):
                        num_of_examples += 1
                        
                        target = toks[ind]
                        mask_token_index = torch.where(input_dict['input_ids'][ind,:]==mask_token_id)[0].detach().cpu().tolist()[0]
                        
                        topk_answers = {}
                        for top_k in ['_top_1', '_top_5', '_top_10', '_top_50', '_top_100']:
                            k = int(top_k.split('_')[-1])
                            topk_answers[top_k] = torch.topk(log_probs[ind, mask_token_index,:], k)[1].tolist()
                            
                            if set(target).intersection(set(topk_answers[top_k])) != set():
                                lm_acc[top_k] += 1
                            
                for top_k in ['_top_1', '_top_5', '_top_10', '_top_50', '_top_100']:
                    lm_acc[top_k] /=num_of_examples
                    final_df[intervention_type+top_k].append(np.round(lm_acc[top_k],3))
            final_df['lang'].append(lang)
    
    if args.num_classifiers>0:
        hook_pos.remove()
        hook_neg.remove()

    final_df = pd.DataFrame(final_df)
    final_df['iters'] = args.num_classifiers
    final_df['langs'] = args.langs
    final_df['model'] = args.model_name_or_path
    final_df['alpha'] = args.alpha
    
    pd.DataFrame(final_df).to_csv(args.output_dir + '/lmacc_langs=' + args.langs + '_model=' + args.model_name_or_path +'_iters=' + str(args.num_classifiers) +'_alpha=' + str(args.alpha) + '_inlp_rand=' + str(args.random_inlp) + '.csv', index=False)

if __name__ == "__main__":
    main()