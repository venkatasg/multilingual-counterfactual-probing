'''
Train iNLP classifier on tokens
'''

import numpy as np
import random
import ipdb
import argparse
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from inlp import debias
    

def run_inlp(num_classifiers, x_train, y_train, x_dev, y_dev, seed):
    '''
    Main function that calls into inlp methods 
    '''
    
    input_dim = x_train.shape[1]
        
    # Define classifier here
    clf = LinearSVC
    params = {"max_iter": 10000, "random_state": seed, "dual": False, "fit_intercept": False}
    
    _, _, Ws_rand, accs_rand = debias.get_random_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=768, is_autoregressive=True, min_accuracy=0, X_dev=x_dev, Y_dev=y_dev, by_class = False)
    
    _, _, Ws, accs = debias.get_debiasing_projection(classifier_class=clf, cls_params=params, num_classifiers=num_classifiers, input_dim=768, is_autoregressive=True, min_accuracy=0, X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, by_class = False)
    
    print("Classifier accuracies:", accs)
    print("Random accuracies:", accs_rand)
    
    return Ws, Ws_rand


def main():
    description = 'Run inlp on data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--num_classifiers', 
        type=int, 
        default=8, 
        help='Number of inlp directions'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help='Random seed'
    )
    parser.add_argument(
        "--langs", 
        type=str, 
        required=True, 
        help="Language pairs on which to train language classifier."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-multilingual-cased',
        help="bert-base-multilingual-cased or xlm-roberta-base",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="mono/reps_",
        help="folder containing hidden states"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mono/reps_inlp/",
        help="folder to output inlp weights to"
    )


    args = parser.parse_args()
    
    # Set random seeds for reproducibility on a specific machine
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)
    
    langs = [x.upper() for x in args.langs.split('_')]
    ids = [i for i in range(len(langs))]
    lang_to_id = {l: i for i, l in enumerate(langs)}
    id_to_lang = {i: l for i, l in enumerate(langs)}
    lang_1 = id_to_lang[0]
    lang_2 = id_to_lang[1]
    
    data_path = args.data_path + args.langs + '/' + args.model_name_or_path + '/'
    
    # Gather arrays
    reps_train_1 = np.load(data_path  + lang_1 + '_train.npy')
    reps_dev_1 = np.load(data_path + lang_1 + '_val.npy')
    
    reps_train_2 = np.load(data_path + lang_2 + '_train.npy')
    reps_dev_2 = np.load(data_path + lang_2 + '_val.npy')
    
    max_train_size = min(reps_train_1.shape[0], reps_train_2.shape[0])
    max_dev_size = min(reps_dev_1.shape[0], reps_dev_2.shape[0])

    # Make train and eval samples
    indices_sample_1 = np.random.choice(reps_train_1.shape[0],max_train_size,replace=False)
    sample_train_1 = reps_train_1[indices_sample_1]
    
    indices_sample_2 = np.random.choice(reps_train_2.shape[0],max_train_size,replace=False)
    sample_train_2 = reps_train_2[indices_sample_2]
    
    indices_sample_dev_1 = np.random.choice(reps_dev_1.shape[0],max_dev_size,replace=False)
    sample_dev_1 = reps_dev_1[indices_sample_dev_1]
    
    indices_sample_dev_2 = np.random.choice(reps_dev_2.shape[0],max_dev_size,replace=False)
    sample_dev_2 = reps_dev_2[indices_sample_dev_2]
    
    # Make X and Y arrays
    y_train = np.array([0 for i in range(sample_train_1.shape[0])] + [1 for i in range(sample_train_2.shape[0])])
    x_train = np.concatenate((sample_train_1, sample_train_2), 0)
    
    y_dev = np.array([0 for i in range(sample_dev_1.shape[0])] + [1 for i in range(sample_dev_2.shape[0])])
    x_dev = np.concatenate((sample_dev_1, sample_dev_2), 0)
    
    # Shuffle all the arrays
    x_train, y_train = shuffle(x_train, y_train, random_state=args.seed)
    x_dev, y_dev = shuffle(x_dev, y_dev, random_state=args.seed)
    
    # Run INLP for each POS classification
    Ws, Ws_rand = run_inlp(num_classifiers=args.num_classifiers, x_train=x_train, y_train=y_train,x_dev=x_dev,y_dev=y_dev, seed=args.seed)
    
    Ws = np.concatenate(Ws)    
    Ws_rand = np.concatenate(Ws_rand)       
    
    output_path = args.output_path
        
    with open(output_path + "Ws.langs={}.model={}.total_iters={}.npy".format(args.langs, args.model_name_or_path, args.num_classifiers), "wb") as f:
        np.save(f, Ws)
        
    # Random parameters
    with open(output_path + "Ws.langs={}.model={}.total_iters={}.rand.npy".format(args.langs, args.model_name_or_path, args.num_classifiers), "wb") as f:
        np.save(f, Ws_rand)

if __name__=="__main__":
    main()