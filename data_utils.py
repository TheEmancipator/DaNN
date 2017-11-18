#!/usr/bin/env python
# -*- coding: utf-8 -*-

# data loader for discourse analyzed datasets
# Author: Sanggyu Han (hsg1991@kaist.ac.kr) & Brian Kangwook Lee (chaximeer@kaist.ac.kr)
# Based on "RST(Rhetorical Structure Theory) tree reader"

import os.path
import nn_inputs
import random
import csv
import sys

SEED = 7753

HOME_PATH = '/home/sghan/project/DaRNN/darnn/'

# call a appropriate method to load the target: Supported dataset list - {tiny_dataset, cornell, imdb, sarcasm}
# Input: a flag for distinguishing the target dataset
def read_discourse_dataset(dataset = 'tiny_dataset', discourse_role = True, type_granularity = 2):
    dataset_path = HOME_PATH + 'dataset/'

    random.seed(SEED)

    if dataset == 'tiny_dataset':
        return _read_tiny_dataset(dataset_path + 'tiny_dataset', discourse_role, type_granularity)
    elif dataset == 'cornell':
        return _read_dataset(dataset_path + 'cornell', discourse_role, type_granularity)
    elif dataset == 'imdb':
        return _read_imdb_dataset(dataset_path + 'imdb', discourse_role, type_granularity)
    elif dataset == 'sarcasm':
        return _read_dataset(dataset_path + 'sarcasm', discourse_role, type_granularity)
    else:
        raise Exception('not supported dataset')

# read the dataset for testing the implementation
# Input: 'tiny dataset(preprocessed by a discourse parser) path'
def _read_tiny_dataset(data_dir, discourse_role, type_granularity):

    # make directories' name for train, dev, test data
    train_dir = os.path.join(data_dir, 'train')
    dev_dir = os.path.join(data_dir, 'dev')
    test_dir = os.path.join(data_dir, 'test')

    names = ['train', 'dev', 'test']
    dir = [train_dir, dev_dir, test_dir]

    data = {}
    overall_max_degree = 0

    for name, sub_dir in zip(names, dir):
        max_degree, trees = _read_trees_tiny(sub_dir, discourse_role, type_granularity)
        data[name] = [(tree, tree.label) for tree in trees]
        overall_max_degree = max(overall_max_degree, max_degree)

    data['max_degree'] = overall_max_degree
    assert overall_max_degree == 2

    print '# train: ' + str(len(data['train']))
    print '# dev: ' + str(len(data['dev']))
    print '# test: ' + str(len(data['test']))

    return data

# read IMDb dataset
# Input: 'IMDb dataset(preprocessed by a discourse parser) path'
# IMDb dataset: http://ai.stanford.edu/~amaas/data/sentiment/
def _read_imdb_dataset(data_dir, discourse_role, type_granularity):

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    names = ['train', 'test']
    dir = [train_dir, test_dir]

    data = {}
    overall_max_degree = 0

    for name, sub_dir in zip(names, dir):
        max_degree, trees = _read_trees_sentiment(sub_dir, discourse_role, type_granularity)
        data[name] = [(tree, tree.label) for tree in trees]
        overall_max_degree = max(overall_max_degree, max_degree)

    # suffle the training data and split them into the development set (10%) and the train set (90%) while keeping the pos/neg ratio (50:50)
    pos_train_data_list = data['train'][:12500]
    neg_train_data_list = data['train'][12500:]
    random.shuffle(pos_train_data_list)
    random.shuffle(neg_train_data_list)
    data['dev'] = pos_train_data_list[:1250] + neg_train_data_list[:1250]
    data['train'] = pos_train_data_list[1250:] + neg_train_data_list[1250:]
    random.shuffle(data['dev'])
    random.shuffle(data['train'])
    random.shuffle(data['test'])

    # deallocate memory
    temp = []
    pos_train_data_list = []
    neg_train_data_list = []

    data['max_degree'] = overall_max_degree
    assert overall_max_degree == 2

    # tester
    print '# train: ' + str(len(data['train']))
    print '# dev: ' + str(len(data['dev']))
    print '# test: ' + str(len(data['test']))

    # final data ratio = training (22,500) / dev (2,500) / test (25,000)
    return data


# read dataset (in this time, Cornell movie review dataset and sarcasm corpus)
# Input: 'dataset path (preprocessed by a discourse parser)'
# Cornell movie review dataset: https://www.cs.cornell.edu/people/pabo/movie-review-data/
# sarcasm corpus: http://storm.cis.fordham.edu/~filatova/SarcasmCorpus.html
def _read_dataset(data_dir, discourse_role, type_granularity):

    dataset = os.path.basename(data_dir)
    data = {}

    if dataset == 'cornell':
        overall_max_degree, trees = _read_trees_sentiment(data_dir, discourse_role, type_granularity)
        temp_data = [(tree, tree.label) for tree in trees]
        # suffle the data and split them into 10 folds while keeping the class ratio (50:50)
        pos_data_list = temp_data[:1000]
        neg_data_list = temp_data[1000:]
        random.shuffle(pos_data_list)
        random.shuffle(neg_data_list)
        for i in range(10):
            data[str(i)] = pos_data_list[(100*i):(100*(i+1))] + neg_data_list[(100*i):(100*(i+1))]
            random.shuffle(data[str(i)])
        # tester
        print '# folds: ' + str(len(data))
        print str(len(data['0'])) + ' ' + str(len(data['1'])) + ' ' + str(len(data['2'])) + ' ' + \
            str(len(data['3'])) + ' ' + str(len(data['4'])) + ' ' + str(len(data['5'])) + ' ' + \
            str(len(data['6'])) + ' ' + str(len(data['7'])) + ' ' + str(len(data['8'])) + ' ' + \
            str(len(data['9']))
        # deallocate memory
        pos_data_list = []
        neg_data_list = []
        temp_data = []
    elif dataset == 'sarcasm':
        overall_max_degree, trees = _read_trees_sarcasm(data_dir, discourse_role, type_granularity)
        temp_data = [(tree, tree.label) for tree in trees]
        iro_data_list = temp_data[:437]
        reg_data_list = temp_data[437:]
        random.shuffle(iro_data_list)
        random.shuffle(reg_data_list)
        for i in range(10):
            if i < 7:
                data[str(i)] = iro_data_list[(44*i):(44*(i+1))] + reg_data_list[(82*i):(82*(i+1))]
                random.shuffle(data[str(i)])
            else:
                data[str(i)] = iro_data_list[(308 + 43*(i-7)):(308 + 43*(i-6))] + reg_data_list[(574 + 81*(i-7)):(574 + 81*(i-6))]
                random.shuffle(data[str(i)])
        # tester
        print '# folds: ' + str(len(data))
        print str(len(data['0'])) + ' ' + str(len(data['1'])) + ' ' + str(len(data['2'])) + ' ' + \
            str(len(data['3'])) + ' ' + str(len(data['4'])) + ' ' + str(len(data['5'])) + ' ' + \
            str(len(data['6'])) + ' ' + str(len(data['7'])) + ' ' + str(len(data['8'])) + ' ' + \
            str(len(data['9']))
        # deallocate memory
        iro_data_list = []
        reg_data_list = []
        temp_data = []
    else:
        raise Exception('not supported dataset')

    data['max_degree'] = overall_max_degree
    assert overall_max_degree == 2

    # final data ratio = training (22,500) / dev (2,500) / test (25,000)
    return data

# read trees in the tiny dataset
# Input: 'sub-directory path in the dataset'
def _read_trees_tiny(sub_dir, discourse_role, type_granularity):

    discourse_type_dict = get_discourse_type_dict(type_granularity)

    files = os.listdir(sub_dir)

    n_files = 0
    trees = []
    max_degree = 0

    for fname in files:
        # ignore trash files (i.e. *.swp)
        if not fname.endswith('.txt'):
            continue

        # count the number of files
        n_files += 1
        # get label from file name
        fname_core = fname.strip().split('.')[0]
        label = int(fname_core.split('_')[1])

        # check labels on file name
        possible_labels = [1,2,3,4,7,8,9,10]
        if not label in possible_labels:
            raise Exception(fname + ' has wrong file name')

        if label>6: # positive
            label=1
        else: # negative
            label=0

        cur_max_degree, cur_tree = _read_tree(os.path.join(sub_dir, fname), discourse_role)
        cur_tree.label = label
        max_degree = max(max_degree, cur_max_degree)
        _remap_discourse_type(cur_tree, discourse_type_dict)
        trees.append(cur_tree)

    trees = sorted(trees, key=lambda k: k.file_name)

    return max_degree, trees

# read trees in the sentiment dataset (i.e. Cornell movie review dataset and IMDb dataset)
# Input: 'sub-directory path in the dataset'
def _read_trees_sentiment(sub_dir, discourse_role, type_granularity):

    discourse_type_dict = get_discourse_type_dict(type_granularity)

    trees = []
    max_degree = 0
    pos_trees = []
    neg_trees = []
    # positive documents
    files = os.listdir(os.path.join(sub_dir, 'pos'))
    for fname in files:
        # ignore trash files (i.e. *.swp)
        if not fname.endswith('.txt'):
            continue
        cur_max_degree, cur_tree = _read_tree(os.path.join(sub_dir, 'pos', fname), discourse_role)
        cur_tree.label = 1 # label = positive
        max_degree = max(max_degree, cur_max_degree)
        _remap_discourse_type(cur_tree, discourse_type_dict)
        pos_trees.append(cur_tree)

    # negative documents
    files = os.listdir(os.path.join(sub_dir, 'neg'))
    for fname in files:
        # ignore trash files (i.e. *.swp)
        if not fname.endswith('.txt'):
            continue
        cur_max_degree, cur_tree = _read_tree(os.path.join(sub_dir, 'neg', fname), discourse_role)
        cur_tree.label = 0 # label = negative
        max_degree = max(max_degree, cur_max_degree)
        _remap_discourse_type(cur_tree, discourse_type_dict)
        neg_trees.append(cur_tree)
    pos_trees = sorted(pos_trees, key=lambda k: k.file_name)
    neg_trees = sorted(neg_trees, key=lambda k: k.file_name)

    trees = pos_trees + neg_trees

    return max_degree, trees

# read trees in the sarcasm corpus
# Input: 'sub-directory path in the dataset'
def _read_trees_sarcasm(sub_dir, discourse_role, type_granularity):

    discourse_type_dict = get_discourse_type_dict(type_granularity)

    trees = []
    max_degree = 0
    iro_trees = []
    reg_trees = []

    # irony documents
    files = os.listdir(os.path.join(sub_dir, 'iro'))
    for fname in files:
        # ignore trash files (i.e. *.swp)
        if not fname.endswith('.txt'):
            continue
        cur_max_degree, cur_tree = _read_tree(os.path.join(sub_dir, 'iro', fname), discourse_role)
        cur_tree.label = 1 # label = irony
        max_degree = max(max_degree, cur_max_degree)
        _remap_discourse_type(cur_tree, discourse_type_dict)
        iro_trees.append(cur_tree)

    # regular documents
    files = os.listdir(os.path.join(sub_dir, 'reg'))
    for fname in files:
        # ignore trash files (i.e. *.swp)
        if not fname.endswith('.txt'):
            continue
        cur_max_degree, cur_tree = _read_tree(os.path.join(sub_dir, 'reg', fname), discourse_role)
        cur_tree.label = 0 # label = regular
        max_degree = max(max_degree, cur_max_degree)
        _remap_discourse_type(cur_tree, discourse_type_dict)
        reg_trees.append(cur_tree)

    iro_trees = sorted(iro_trees, key=lambda k: k.file_name)
    reg_trees = sorted(reg_trees, key=lambda k: k.file_name)

    trees = iro_trees + reg_trees

    return max_degree, trees

def _read_tree(fname, discourse_role):
    file_discourse = open(fname)
    discourse = file_discourse.readlines()[0]

    # check discourse included in file
    if len(discourse) == 0:
        raise Exception(fname + ' includes wrong discourse')

    k, tree, max_degree = _parse(discourse, "root", discourse_role, 0)
    assert discourse.count('edu') == tree.num_leaves

    # store the file name at the root node
    tree.file_name = os.path.basename(fname)

    # Treat special case - only root node exists
    # Duplicate same root node as children (node_num: 1->3)
    if len(tree.children)==0:
        # print 'DEBUG: data_utils.py'
        tree.discourse_role = 'NN'
        for i in range(2):
            node = nn_inputs.Node()

            # TODO:exp10dot5
            # node.discourse_relation_type = 'sole_edu'
            node.discourse_relation_type = 'root'

            node.val = tree.val
            node.label = None
            tree.add_child(node)
        tree.val = None

        #TODO:exp10dot5
        # tree.discourse_relation_type = 'root'
        tree.discourse_relation_type = 'sole_edu'


    return max_degree, tree


def _parse(text, discourse_relation_type, discourse_role, i):

    # skip indents, new line, right_child parenthesis and space
    while text[i] != '(':
        i+=1

    i+=1
    token = text[i:].split(' ')[0]
    i+=(len(token)+1)

    node = nn_inputs.Node()
    # node.discourse_relation_type = discourse_relation_type
    node.label = None

    if token=='EDU':
        k = text.find(')', i)
        node.val = int(text[i+3:].split(')')[0])
        max_degree = 0

        #TODO:exp10dot5
        node.discourse_relation_type = 'root'

    else:
        node.discourse_role = token.split('-')[1][0:2]
        child_discourse_relation_type = token.split('-')[0]

        #TODO:exp10dot5
        node.discourse_relation_type = child_discourse_relation_type

        if not (node.discourse_role[0]=='N' or node.discourse_role[0]=='S'):
            raise Exception('unseen case')

        k, left_child, left_child_max_degree = _parse(text, child_discourse_relation_type, discourse_role, i)
        k, right_child, right_child_max_degree = _parse(text, child_discourse_relation_type, discourse_role, k)
        if discourse_role == True:  # reorder the children nodes by their discourse role
            if node.discourse_role == 'SN':
                node.add_child(right_child)
                node.add_child(left_child)
            else:
                node.add_child(left_child)
                node.add_child(right_child)
        else:
            node.add_child(left_child)
            node.add_child(right_child)

        
        cur_max_degree = len(node.children)
        max_degree = max(cur_max_degree, left_child_max_degree, right_child_max_degree)

        # in the case of RST parse tree, # children is alway 0 or 2
        assert len(node.children)==0 or len(node.children)==2

    return k, node, max_degree

def _remap_tokens_and_labels(tree, sentence, fine_grained):
    # map leaf idx to word idx
    if tree.val is not None:
        tree.val = sentence[tree.val]

    [_remap_tokens_and_labels(child, sentence, fine_grained)
     for child in tree.children
     if child is not None]

def _remap_discourse_type(tree, discourse_type_dict):
    # map discourse_type_string to discourse_type_idx
    if tree.discourse_relation_type is None:
        raise Exception('tree.discourse_relation_type has no value')

    if tree.discourse_relation_type == 'root':
        tree.discourse_relation_type = -1
    else:
        tree.discourse_relation_type = discourse_type_dict[tree.discourse_relation_type]

    [_remap_discourse_type(child, discourse_type_dict)
     for child in tree.children
     if child is not None]


# def get_discourse_type_dict():
#     discourse_type_list_fname = HOME_PATH + 'dataset/discourse_relation_type_list.txt'
#     discourse_type_list_file = open(discourse_type_list_fname)
#     discourse_type_dict = {}
#     discourse_type_idx = 0
#     for discourse_type in discourse_type_list_file.readlines():
#         discourse_type_dict[discourse_type.strip()] = discourse_type_idx
#         discourse_type_idx += 1
#     return discourse_type_dict


def get_discourse_type_dict(type_granularity):
    discourse_type_list_fname = HOME_PATH + 'dataset/discourse_relation_type_list.txt'
    discourse_type_list_file = open(discourse_type_list_fname)
    discourse_type_dict = {}
    discourse_type_idx = 0
    for discourse_type in discourse_type_list_file.readlines():
        discourse_type_dict[discourse_type.strip()] = discourse_type_idx
        discourse_type_idx += 1

    if type_granularity != 2:
        csv_fname = HOME_PATH + 'dataset/discourse_relation_type_definitions.csv'
        csv_file = open(csv_fname)
        reader = csv.reader(csv_file, delimiter=',')

        dict_type2cluster = {}
        dict_cluster2idx = {}
        cluster_idx = 0
        for row in reader:
            dict_type2cluster[row[2].strip()] = row[type_granularity].strip()

            if not (row[type_granularity].strip() in dict_cluster2idx):
                dict_cluster2idx[row[type_granularity].strip()] = cluster_idx
                cluster_idx += 1

        for i in discourse_type_dict:
            discourse_type_dict[i] = dict_cluster2idx[dict_type2cluster[i]]

    # error check
    num_discourse_type = len(set(discourse_type_dict.values()))
    print type_granularity
    print num_discourse_type
    if type_granularity==0:
        # assert num_discourse_type == 4
        assert num_discourse_type == 2
    elif type_granularity==1:
        assert num_discourse_type == 17
    elif type_granularity==2:
        assert num_discourse_type == 35
    else:
        raise Exception('data_utils.py: type_granularity has wrong value')

    return discourse_type_dict

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data = read_discourse_dataset(dataset=DATA_DIR, discourse_role=False)
    if DATA_DIR == 'sarcasm' or DATA_DIR == 'cornell':
        print data['0'][0][0].file_name
        print data['0'][1][0].file_name
        print data['0'][2][0].file_name
        print data['0'][3][0].file_name
        print data['0'][4][0].file_name
    else:
        print data['dev'][0][0].file_name
        print data['dev'][1][0].file_name
        print data['dev'][2][0].file_name
        print data['dev'][3][0].file_name
        print data['dev'][4][0].file_name
