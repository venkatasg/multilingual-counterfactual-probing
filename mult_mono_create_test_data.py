import argparse
import pandas as pd
import ipdb
from collections import Counter
import random
from transformers import AutoTokenizer, set_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="R"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Location of input data to create masked data. Masked data is stored in same directory",
        required=True
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Two possible values - test or val.Create test data or validation data",
        required=True
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="bert-base-multilingual-cased or xlm-roberta-base",
        default='bert-base-multilingual-cased'
    )
    parser.add_argument(
        "--seed",
        type=int, 
        default=1, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--langs", 
        type=str, 
        required=True, 
        help="language pairs on which to train language classifier."
    )
    
    args = parser.parse_args()
    
    return args

def parse_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n\n')
        examples = []
        for example in lines:
            example = example.split('\n')
            words = [line.split('\t')[0] for line in example]
            labels = [line.split('\t')[-1] for line in example]
            examples.append({'words': words, 'lang': labels[0]})
    examples = pd.DataFrame(examples).reset_index(drop=True)
    return examples
    
def sample_word(word_list, tokenizer, num_of_toks):
    sampled_random = False
    while sampled_random==False:
        sampled_word = random.sample(word_list, 1)[0]
        num_toks_random = len(tokenizer(sampled_word, add_special_tokens=False)['input_ids'])
        if num_toks_random==num_of_toks:
            sampled_random = True
    return sampled_word
    
def return_masked_sents_tuple(row, words, lang_dict, tokenizer, control_words):

    sent_words = row['words']
    words_in_sent = [a for a in sent_words if a in words]
    
    # If the sampled word is not in the translation dictionary move on
    if len(words_in_sent) == 0:
        return None
    
    true_word = random.sample(list(words_in_sent), 1)[0]
    transl_word = lang_dict[true_word]
    
    # Find number of tokens it gets split into
    num_toks_true = len(tokenizer(true_word, add_special_tokens=False)['input_ids'])
    num_toks_transl = len(tokenizer(transl_word, add_special_tokens=False)['input_ids'])
    
    # Sample a random word in true and transl language
    true_randoms = [a for a in lang_dict.keys() if a]
    true_random = sample_word(true_randoms, tokenizer, num_toks_true)
    
    transl_randoms = [lang_dict[a] for a in lang_dict.keys() if (a and lang_dict[a])]
    transl_random = sample_word(transl_randoms, tokenizer, num_toks_transl)
    
    # Sample a control token from some other language that has same number of tokens as true answer and transl answer
    true_control = sample_word(control_words, tokenizer, num_toks_true)
    transl_control = sample_word(control_words, tokenizer, num_toks_transl)
    
    # Now make the masked sentence    
    mask_token = tokenizer.mask_token    
    masked_sentence = sent_words
    
    for tok_index, tok in enumerate(masked_sentence):
        if tok==true_word:
            mask_tokens_to_insert_true = [mask_token for i in range(num_toks_true)]
            masked_sentence_true = masked_sentence[:tok_index] + mask_tokens_to_insert_true + masked_sentence[tok_index+1:]
            
            mask_tokens_to_insert_transl = [mask_token for i in range(num_toks_transl)]
            masked_sentence_transl = masked_sentence[:tok_index] + mask_tokens_to_insert_transl + masked_sentence[tok_index+1:]
            break
    
    return (masked_sentence_true, masked_sentence_transl, true_word, transl_word, true_control, transl_control, true_random, transl_random)

def write_to_file(filename, sent_list):
    with open(filename, 'w') as f:
        f.write("masked_sent_true\tmasked_sent_transl\ttrue_answer\ttransl_answer\ttrue_control\ttransl_control\ttrue_random\ttransl_random\n")
        for (masked_sent_true, masked_sent_transl, true_answer, transl_answer, true_control, transl_control, true_random, transl_random) in sent_list:
            f.write(" ".join(masked_sent_true) + "\t" + " ".join(masked_sent_transl) + "\t" + true_answer + "\t" + transl_answer + "\t" + true_control +"\t" + transl_control + "\t" + true_random + "\t" + transl_random + "\n")

def main():
    # Parse all args
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    random.seed(1)
    
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    
    lang_1 = args.langs.split('_')[0]
    lang_2 = args.langs.split('_')[1]
    
    ###### Prepare the dataset with masks in certain locations and correct answer known #########
    
    test_df = parse_data(args.input_path + args.test + '.txt')

    lang1_df = test_df.loc[test_df['lang']==lang_1.upper(), :].reset_index(drop=True)
    lang2_df = test_df.loc[test_df['lang']==lang_2.upper(), :].reset_index(drop=True)
    
    lang12_dict = {}
    lang21_dict = {}
    
    with open('data/muse/' + args.langs + '-filtered.txt') as f:
        for line in f.readlines():
            lang12_dict[line.split('\t')[0].strip()] = line.split('\t')[1].strip()
            lang21_dict[line.split('\t')[1].strip()] = line.split('\t')[0].strip()
    
    lang1_words = list(lang12_dict.keys())
    lang2_words = list(lang21_dict.keys())
    
    # Make a list of control words to sample from
    control_dict = {'en_hi': ['en_ko', 'en_fi', 'en_es'], 'en_ko': ['en_fi', 'en_hi', 'en_es'], 'en_es': ['en_ko', 'en_hi', 'en_fi'], 'en_fi': ['en_hi', 'en_ko', 'en_es']}
    control_langs = control_dict[args.langs]
    control_words = []
    for lang in control_langs:
        with open('data/muse/' + lang + '-filtered.txt') as f:
            for line in f.readlines():
                control_words.append(line.split('\t')[1].strip())
        
    lang1_df['mask_dict'] = lang1_df.apply(lambda row: return_masked_sents_tuple(row, lang1_words, lang12_dict, tokenizer, control_words), axis=1)
    masked_sentences_1 = [x for x in lang1_df['mask_dict'] if x]
    
    lang2_df['mask_dict'] = lang2_df.apply(lambda row: return_masked_sents_tuple(row, lang2_words, lang21_dict, tokenizer, control_words), axis=1)
    masked_sentences_2 = [x for x in lang2_df['mask_dict'] if x]

    ###################
    
    filename_1_path = args.input_path + lang_1 + '_' + args.model_name_or_path + '-' + args.test + '.tsv'
    write_to_file(filename_1_path, masked_sentences_1)
    
    filename_2_path = args.input_path + lang_2 + '_' + args.model_name_or_path + '-' + args.test + '.tsv'
    write_to_file(filename_2_path, masked_sentences_2)

if __name__ == "__main__":
    main()