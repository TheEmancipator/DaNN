#!/usr/bin/env python
# -*- coding: utf-8 -*-

# An implimentation of the discourse-aware tree-structure neural networks, including RecNN, tree RNN, tree LSTM.
# Author: Brian Kangwook Lee (chaximeer@kaist.ac.kr) & Sanggyu Han (hsg1991@kaist.ac.kr)

import nn_inputs
import recnn
import tree_rnn
import tree_lstm
import tree_gru
import numpy as np
import theano
from theano import tensor as T
import random
import cPickle
from data_utils import read_discourse_dataset
import sys
import time
import socket

theano.config.floatX = 'float32'

EXP_VER = '1' # define the version of the current experiment
HOST_NAME = socket.gethostname()
HOME_PATH = '.' # define the root path of the DaNN

'''
    < discourse_exp6dot5 >

    ::::: Description
    For each 'dataset' or all 'dataset'
    * Discourse_type_granularity is decided at (2) exp7dot5

    For each 'dataset',
    * Optimal ('optimizer', 'lr') is already picked at (1) exp4dot5
    * Optimal ('l2_reg_param', 'dropout_ratio') is already picked at (3) exp5dot5
    * Repeat for 8 models {RecNN, LSTM} x {Role O, Role X} x {Type O, Type X}
    * Proper 'epoch' is empirically chosen

    ::::: Tunable params
    [1] dataset \in {imdb, sarcasm, cornell}
    [2] model \in {RecNN, LSTM}
    [3] role \in {True, False}
    [4] type \in {True, False}
    [5] epoch

    ::::: Dependent params
    Depend on [1] EMB_DIM, HIDDEN_DIM, OPTIMIZER, LEARNING_RATE, L2, DROPOUT_RATIO
    Depend on [2]
    Depend on [3]
    Depend on [4]
    Depend on [5]

    ::::: Fixed params
    [train] train_shuffle  = True
    [model] IS_N_ARY       = True

'''

'''



    Tunable parameter
'''
DATASET = sys.argv[1]
WHICH_CELL = sys.argv[2]
DISCOURSE_ROLE = sys.argv[3]
DISCOURSE_TYPE = sys.argv[4]
'''



    Dependent params
'''

# Substitute proper value / Exception handling
if DATASET == 't':
    DATASET = 'tiny_dataset'
    EMB_DIM = 200
    HIDDEN_DIM = 200
    OPTIMIZER = 's'
    LEARNING_RATE = 0.01
    L2_REG_PARAM = 0
    DROPOUT_RATIO = 0
    NUM_EPOCHS = None
    TYPE_GRANULARITY = 2

elif DATASET == 'i':
    DATASET = 'imdb'
    EMB_DIM = 300
    HIDDEN_DIM = 300
    OPTIMIZER = 'a'
    LEARNING_RATE = 0.00001
    L2_REG_PARAM = 0.0000000001
    DROPOUT_RATIO = 0.5
    NUM_EPOCHS = 25
    TYPE_GRANULARITY = 2

elif DATASET == 'c':
    DATASET = 'cornell'
    EMB_DIM = 100
    HIDDEN_DIM = 100
    OPTIMIZER = 'a'
    LEARNING_RATE = 0.00005
    L2_REG_PARAM = 0.0000000001
    DROPOUT_RATIO = 0.5
    NUM_EPOCHS = 25
    TYPE_GRANULARITY = 2

elif DATASET == 's':
    DATASET = 'sarcasm'
    EMB_DIM = 50
    HIDDEN_DIM = 50
    OPTIMIZER = 'a'
    LEARNING_RATE = 0.0001
    L2_REG_PARAM = 0.0000000001
    DROPOUT_RATIO = 0.4
    NUM_EPOCHS = 30
    TYPE_GRANULARITY = 2

else:
    raise Exception('argv[1] DATASET has wrong value')

'''



    Fixed parameter
'''
EPOCH_TRAIN_SHUFFLE = True
IS_N_ARY = 't'

'''



    Constants
'''
NUM_LABELS = 2
TRAINABLE_EMBEDDING = True
MOMENTUM = 0.9
SEED = 7753
IRREGULAR_TREE = False
ONLY_LEAVES_HAVE_VALS = True # For integrity check (Not params for exp)
MAX_DEGREE = 2
LABELS_ON_NONROOT_NODES = True

# AVAILABLE_DATASET = ['t', 'i', 'c', 's']
# AVAILABLE_WHICH_CELL = ['RecNN', 'RNN', 'LSTM', 'GRU']
# AVAILABLE_DISCOURSE_ROLE = ['t', 'f']
# AVAILABLE_IS_N_ARY = ['t', 'f']
# AVAILABLE_DISCOURSE_TYPE = ['t', 'f']
# AVAILABLE_OPTIMIZER = ['s', 'r', 'a']
# AVAILABLE_L2_REG_PARAM = [0, 0.0001, 0.00001, 0.000001]
# AVAILABLE_DROPOUT_RATIO = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

AVAILABLE_DATASET = ['i', 'c', 's']
AVAILABLE_WHICH_CELL = ['RecNN', 'LSTM', 'GRU']
AVAILABLE_TYPE_GRANULARITY = [0, 1, 2]
AVAILABLE_DISCOURSE_ROLE = ['t', 'f']
AVAILABLE_IS_N_ARY = ['t']
AVAILABLE_DISCOURSE_TYPE = ['t', 'f']
AVAILABLE_OPTIMIZER = ['a']
AVAILABLE_DROPOUT_RATIO = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


if not (WHICH_CELL in AVAILABLE_WHICH_CELL):
    raise Exception('argv WHICH_CELL has wrong value')

if TYPE_GRANULARITY == 0:
    NUM_DISCOURSE_EMB = 4 # Except 'root'
    DISCOURSE_EMB_DIM = 2
elif TYPE_GRANULARITY == 1:
    NUM_DISCOURSE_EMB = 17 # Except 'root'
    DISCOURSE_EMB_DIM = 5
elif TYPE_GRANULARITY == 2:
    NUM_DISCOURSE_EMB = 35 # Except 'root'
    DISCOURSE_EMB_DIM = 10
else:
    raise Exception('argv TYPE_GRANULARITY has wrong value')


if DISCOURSE_ROLE == 't':
    DISCOURSE_ROLE = True
elif DISCOURSE_ROLE == 'f':
    DISCOURSE_ROLE = False
else:
    raise Exception('argv DISCOURSE_ROLE has wrong value')

if IS_N_ARY == 't':
    IS_N_ARY = True
elif IS_N_ARY == 'f':
    IS_N_ARY = False
else:
    raise Exception('IS_N_ARY has wrong value')

if DISCOURSE_TYPE == 't':
    DISCOURSE_TYPE = True
elif DISCOURSE_TYPE == 'f':
    DISCOURSE_TYPE = False
else:
    raise Exception('argv DISCOURSE_TYPE has wrong value')

if not (OPTIMIZER in AVAILABLE_OPTIMIZER):
    raise Exception('argv OPTIMIZER has wrong value')

# if not (L2_REG_PARAM in AVAILABLE_L2_REG_PARAM):
#     raise Exception('argv L2_REG_PARAM has wrong value')

if not (DROPOUT_RATIO in AVAILABLE_DROPOUT_RATIO):
    raise Exception('argv DROPOUT_RATIO has wrong value')


# Print params
print 'DATASET        :', DATASET
print 'OPTIMIZER      :', OPTIMIZER
print 'LEARNING_RATE  :', LEARNING_RATE
print

print 'EMB_DIM        :', EMB_DIM
print 'HIDDEN_DIM     :', HIDDEN_DIM
print

print 'NUM_EPOCHS         :', NUM_EPOCHS
print 'EPOCH_TRAIN_SHUFFLE:', EPOCH_TRAIN_SHUFFLE
print 'WHICH_CELL         :', WHICH_CELL
print 'IS_N_ARY           :', IS_N_ARY
print 'DISCOURSE_ROLE     :', DISCOURSE_ROLE
print 'DISCOURSE_TYPE     :', DISCOURSE_TYPE
print 'L2_REG_PARAM       :', L2_REG_PARAM
print 'DROPOUT_RATIO      :', DROPOUT_RATIO
print

'''



    Tree-XXX Model
'''

class DiscourseModelRecNN(recnn.RecNN):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)

    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_h**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss


class DiscourseModelRNN(tree_rnn.TreeRNN):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)

    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_hx**2)
            l2 += T.sum(self.W_hh**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss

class DiscourseModelNaryRNN(tree_rnn.NaryTreeRNN):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)

    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_hx**2)
            l2 += T.sum(self.W_hh**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss

class DiscourseModelChildSumLSTM(tree_lstm.ChildSumTreeLSTM):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)

    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_i**2)
            l2 += T.sum(self.W_f**2)
            l2 += T.sum(self.W_o**2)
            l2 += T.sum(self.W_u**2)
            l2 += T.sum(self.U_i**2)
            l2 += T.sum(self.U_f**2)
            l2 += T.sum(self.U_o**2)
            l2 += T.sum(self.U_u**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss

class DiscourseModelNaryLSTM(tree_lstm.NaryTreeLSTM):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)


    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_i**2)
            l2 += T.sum(self.W_f**2)
            l2 += T.sum(self.W_o**2)
            l2 += T.sum(self.W_u**2)
            l2 += T.sum(self.U_i**2)
            l2 += T.sum(self.U_f**2)
            l2 += T.sum(self.U_o**2)
            l2 += T.sum(self.U_u**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss

class DiscourseModelChildSumGRU(tree_gru.ChildSumTreeGRU):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)


    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_z**2)
            l2 += T.sum(self.W_r**2)
            l2 += T.sum(self.W_h**2)
            l2 += T.sum(self.U_z**2)
            l2 += T.sum(self.U_r**2)
            l2 += T.sum(self.U_h**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss


class DiscourseModelNaryGRU(tree_gru.NaryTreeGRU):
    if DISCOURSE_TYPE:
        def train_step_inner(self, x, tree, y, y_exists, discourse_x):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists, discourse_x)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0
            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist, discourse_x)
            return loss, origin_loss, pred_y

        def evaluate(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._evaluate(x, tree[:, :-1], discourse_x)

        def predict(self, root_node):
            x, tree, _, _, discourse_x = \
                nn_inputs.gen_nn_inputs_disc_type(root_node             = root_node,
                                                  max_degree            = self.degree,
                                                  only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                                  with_labels           = True)
            self._check_input(x, tree)
            return self._predict(x, tree[:, :-1], discourse_x)


    else:
        def train_step_inner(self, x, tree, y, y_exists):
            self._check_input(x, tree)
            return self._train(x, tree[:, :-1], y, y_exists)

        def train_step(self, root_node, label):
            x, tree, labels, labels_exist = \
                nn_inputs.gen_nn_inputs(root_node             = root_node,
                                        max_degree            = self.degree,
                                        only_leaves_have_vals = ONLY_LEAVES_HAVE_VALS,
                                        with_labels           = True)

            labels[np.logical_not(labels_exist)] = 0
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
            y[np.logical_not(labels_exist)] = 0

            loss, origin_loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
            return loss, origin_loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        train_loss = T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

        if L2_REG_PARAM != 0:
            l2 = T.sum(self.embeddings**2)
            l2 += T.sum(self.W_out**2)
            if DISCOURSE_TYPE:
                l2 += T.sum(self.discourse_embeddings**2)
            l2 += T.sum(self.W_z**2)
            l2 += T.sum(self.W_r**2)
            l2 += T.sum(self.W_h**2)
            l2 += T.sum(self.U_z**2)
            l2 += T.sum(self.U_r**2)
            l2 += T.sum(self.U_h**2)
            reg_train_loss = train_loss + L2_REG_PARAM*l2
            return reg_train_loss, train_loss

        else:
            return train_loss, train_loss

'''



    get_model function
'''
def get_model(num_emb, max_degree):
    if WHICH_CELL == 'RecNN':
        return DiscourseModelRecNN(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                   emb_dim       = EMB_DIM,       # 200
                                   hidden_dim    = EMB_DIM,       # 200 <<-- Check this out!
                                   output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                   degree        = max_degree,    # 2
                                   learning_rate = LEARNING_RATE, # 0.01
                                   momentum      = MOMENTUM,      # 0.9
                                   trainable_embeddings    = TRAINABLE_EMBEDDING,
                                   labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                   irregular_tree          = IRREGULAR_TREE,
                                   optimizer               = OPTIMIZER,
                                   discourse_type          = DISCOURSE_TYPE,
                                   dropout_ratio           = DROPOUT_RATIO,
                                   num_discourse_emb       = NUM_DISCOURSE_EMB,
                                   discourse_emb_dim       = DISCOURSE_EMB_DIM)

    if WHICH_CELL == 'RNN':
        if IS_N_ARY:
            return DiscourseModelNaryRNN(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                         emb_dim       = EMB_DIM,       # 200
                                         hidden_dim    = HIDDEN_DIM,    # 100
                                         output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                         degree        = max_degree,    # 2
                                         learning_rate = LEARNING_RATE, # 0.01
                                         momentum      = MOMENTUM,      # 0.9
                                         trainable_embeddings    = TRAINABLE_EMBEDDING,
                                         labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                         irregular_tree          = IRREGULAR_TREE,
                                         optimizer               = OPTIMIZER,
                                         discourse_type          = DISCOURSE_TYPE,
                                         dropout_ratio           = DROPOUT_RATIO,
                                         num_discourse_emb       = NUM_DISCOURSE_EMB,
                                         discourse_emb_dim       = DISCOURSE_EMB_DIM)
        else:
            return DiscourseModelRNN(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                     emb_dim       = EMB_DIM,       # 200
                                     hidden_dim    = HIDDEN_DIM,    # 100
                                     output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                     degree        = max_degree,    # 2
                                     learning_rate = LEARNING_RATE, # 0.01
                                     momentum      = MOMENTUM,      # 0.9
                                     trainable_embeddings    = TRAINABLE_EMBEDDING,
                                     labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                     irregular_tree          = IRREGULAR_TREE,
                                     optimizer               = OPTIMIZER,
                                     discourse_type          = DISCOURSE_TYPE,
                                     dropout_ratio           = DROPOUT_RATIO,
                                     num_discourse_emb       = NUM_DISCOURSE_EMB,
                                     discourse_emb_dim       = DISCOURSE_EMB_DIM)
    elif WHICH_CELL == 'LSTM':
        if IS_N_ARY:
            return DiscourseModelNaryLSTM(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                          emb_dim       = EMB_DIM,       # 200
                                          hidden_dim    = HIDDEN_DIM,    # 100
                                          output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                          degree        = max_degree,    # 2
                                          learning_rate = LEARNING_RATE, # 0.01
                                          momentum      = MOMENTUM,      # 0.9
                                          trainable_embeddings    = TRAINABLE_EMBEDDING,
                                          labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                          irregular_tree          = IRREGULAR_TREE,
                                          optimizer               = OPTIMIZER,
                                          discourse_type          = DISCOURSE_TYPE,
                                          dropout_ratio           = DROPOUT_RATIO,
                                          num_discourse_emb       = NUM_DISCOURSE_EMB,
                                          discourse_emb_dim       = DISCOURSE_EMB_DIM)
        else:
            return DiscourseModelChildSumLSTM(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                              emb_dim       = EMB_DIM,       # 200
                                              hidden_dim    = HIDDEN_DIM,    # 100
                                              output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                              degree        = max_degree,    # 2
                                              learning_rate = LEARNING_RATE, # 0.01
                                              momentum      = MOMENTUM,      # 0.9
                                              trainable_embeddings    = TRAINABLE_EMBEDDING,
                                              labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                              irregular_tree          = IRREGULAR_TREE,
                                              optimizer               = OPTIMIZER,
                                              discourse_type          = DISCOURSE_TYPE,
                                              dropout_ratio           = DROPOUT_RATIO,
                                              num_discourse_emb       = NUM_DISCOURSE_EMB,
                                              discourse_emb_dim       = DISCOURSE_EMB_DIM)
    elif WHICH_CELL == 'GRU':
        if IS_N_ARY:
            return DiscourseModelNaryGRU(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                         emb_dim       = EMB_DIM,       # 200
                                         hidden_dim    = HIDDEN_DIM,    # 100
                                         output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                         degree        = max_degree,    # 2
                                         learning_rate = LEARNING_RATE, # 0.01
                                         momentum      = MOMENTUM,      # 0.9
                                         trainable_embeddings    = TRAINABLE_EMBEDDING,
                                         labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                         irregular_tree          = IRREGULAR_TREE,
                                         optimizer               = OPTIMIZER,
                                         discourse_type          = DISCOURSE_TYPE,
                                         dropout_ratio           = DROPOUT_RATIO,
                                         num_discourse_emb       = NUM_DISCOURSE_EMB,
                                         discourse_emb_dim       = DISCOURSE_EMB_DIM)
        else:
            return DiscourseModelChildSumGRU(num_emb       = num_emb,       # len(dic_eduidx2embedding)
                                             emb_dim       = EMB_DIM,       # 200
                                             hidden_dim    = HIDDEN_DIM,    # 100
                                             output_dim    = NUM_LABELS,    # NUM_LABELS=2
                                             degree        = max_degree,    # 2
                                             learning_rate = LEARNING_RATE, # 0.01
                                             momentum      = MOMENTUM,      # 0.9
                                             trainable_embeddings    = TRAINABLE_EMBEDDING,
                                             labels_on_nonroot_nodes = LABELS_ON_NONROOT_NODES,
                                             irregular_tree          = IRREGULAR_TREE,
                                             optimizer               = OPTIMIZER,
                                             discourse_type          = DISCOURSE_TYPE,
                                             dropout_ratio           = DROPOUT_RATIO,
                                             num_discourse_emb       = NUM_DISCOURSE_EMB,
                                             discourse_emb_dim       = DISCOURSE_EMB_DIM)
    else:
        raise Exception('Not supported model')


def train():

    train_set = None
    dev_set = None
    test_set = None
    max_degree = None

    # Exception cases (imdb+ten_fold or tiny_dataset+ten_fold)
    if DATASET == 'imdb' or DATASET == 'tiny_dataset':
        # Load discourse dataset
        data = read_exp1_dataset(DATASET)

        train_set, dev_set, test_set = data['train'], data['dev'], data['test']
        max_degree = data['max_degree']
        print 'train', len(train_set)
        print 'dev', len(dev_set)
        print 'test', len(test_set)
        print 'max degree', max_degree
        print 'num labels', NUM_LABELS

        # Loaded_data integrity check
        for key, dataset in data.items():
            if key == 'max_degree':
                continue
            labels = [label for _, label in dataset]
            assert set(labels) <= set(xrange(NUM_LABELS)), set(labels)

        # Set random seed
        random.seed(SEED)
        np.random.seed(SEED)

        # Load word embeddings
        f = open(HOME_PATH + 'embedding/' + DATASET + '.pkl')
        dic_eduidx2embedding = cPickle.load(f)
        num_emb = len(dic_eduidx2embedding)

        # Make model
        model = get_model(num_emb, max_degree)

        # initialize model embeddings to glove
        embeddings = model.embeddings.get_value()
        for edu_idx, real_vec in dic_eduidx2embedding.iteritems():
            embeddings[int(edu_idx[3:])] = real_vec
        dic_eduidx2embedding = [], [], []

        f_exp, f_answersheet_dev, f_answersheet_test = prepare_result_files()

        # Set embedding value
        model = get_model(num_emb, max_degree)
        model.embeddings.set_value(embeddings)

        dev_answer, test_answer, _, _, _ = train_fold(f_exp, model, train_set, dev_set, test_set)
        write_answersheet(f_answersheet_dev, dev_set, dev_answer)
        write_answersheet(f_answersheet_test, test_set, test_answer)

    else:
        pre_loaded_data = read_discourse_dataset(DATASET, DISCOURSE_ROLE, TYPE_GRANULARITY)
        max_degree = MAX_DEGREE

        # Set random seed
        random.seed(SEED)
        np.random.seed(SEED)

        # Load word embeddings
        f = open(HOME_PATH + 'embedding/' + DATASET + '.pkl')
        dic_eduidx2embedding = cPickle.load(f)
        num_emb = len(dic_eduidx2embedding)

        # Make model
        model = get_model(num_emb, max_degree)

        # initialize model embeddings to glove
        embeddings = model.embeddings.get_value()
        for edu_idx, real_vec in dic_eduidx2embedding.iteritems():
            embeddings[int(edu_idx[3:])] = real_vec
        dic_eduidx2embedding = [], [], []

        f_exp, f_answersheet_dev, f_answersheet_test = prepare_result_files()

        train_loss_list = []
        dev_score_list = []
        test_score_list = []

        for fold in range(10):
            # Set embedding value
            model = get_model(num_emb, max_degree)
            model.embeddings.set_value(embeddings)

            dev_fold = fold
            test_fold = (dev_fold + 1) % 10

            data = read_ten_fold_dataset(pre_loaded_data,
                                         dev_fold=dev_fold,
                                         test_fold=test_fold)
            train_set, dev_set, test_set = data['train'], data['dev'], data['test']

            print
            print '**TEST_FOLD:', test_fold

            f_exp.write('TEST_FOLD:%d\n' % test_fold)
            f_answersheet_dev.write('DEV_FOLD:%d\n' % dev_fold)
            f_answersheet_test.write('TEST_FOLD:%d\n' % test_fold)
            dev_answer, test_answer, train_loss, dev_score, test_score = train_fold(f_exp, model, train_set, dev_set, test_set)

            train_loss_list.append(train_loss)
            dev_score_list.append(dev_score)
            test_score_list.append(test_score)

            write_answersheet(f_answersheet_dev, dev_set, dev_answer)
            write_answersheet(f_answersheet_test, test_set, test_answer)

        train_loss_mean = np.mean(train_loss_list)
        dev_score_mean = np.mean(dev_score_list)
        test_score_mean = np.mean(test_score_list)
        f_exp.write('10-fold Avg Score, %.15f, %.15f, %.15f\n' % (train_loss_mean, dev_score_mean, test_score_mean))

    print 'finished training'
    f_exp.write('\n')
    f_answersheet_dev.write('\n')
    f_answersheet_test.write('\n')
    f_exp.close()
    f_answersheet_dev.close()
    f_answersheet_test.close()


def train_fold(f_exp, model, train_set, dev_set, test_set):
    start_time = time.time()
    train_loss_list = []
    dev_score_list = []
    test_score_list = []
    dev_answer_list = []
    test_answer_list = []
    f_exp.write('epoch, avg_loss, dev_score, test_score, sec\n')
    for epoch in xrange(NUM_EPOCHS):

        if EPOCH_TRAIN_SHUFFLE:
            random.shuffle(train_set)

        print 'epoch', epoch
        avg_loss, avg_reg_loss = train_dataset(model, train_set)
        print 'avg loss', avg_loss
        print 'avg reg loss', avg_reg_loss
        dev_score = evaluate_dataset(model, dev_set)
        print 'dev score', dev_score
        test_score = evaluate_dataset(model, test_set)
        print 'test score', test_score
        elapsed_time = time.time() - start_time
        print '%s seconds until epoch %d' % (elapsed_time, epoch)
        f_exp.write('%d, %f, %f, %f, %f\n' % (epoch, avg_loss, dev_score, test_score, elapsed_time))

        # For obtaining Best score on dev set
        train_loss_list.append(avg_loss)
        dev_score_list.append(dev_score)
        test_score_list.append(test_score)

        dev_answer_list.append(get_answer_each_epoch(model, dev_set))
        test_answer_list.append(get_answer_each_epoch(model, test_set))

    # Obtain Best score w/ dev_set -> train_set -> epoch
    train_loss_array = np.array(train_loss_list)
    dev_score_array = np.array(dev_score_list)

    best_dev_score = np.max(dev_score_array)
    best_dev_score_idx = np.where(dev_score_array==best_dev_score)
    
    # 1st tie break
    if len(best_dev_score_idx[0])==1:
        best_idx = best_dev_score_idx[0][0]
    else:
        selected_train_loss_array = train_loss_array[best_dev_score_idx]
        best_train_loss = np.min(selected_train_loss_array)
        best_train_loss_idx = np.where(train_loss_array==best_train_loss)

        train_tie_break = np.intersect1d(best_dev_score_idx, best_train_loss_idx)

        # 2nd tie break
        if len(train_tie_break)==1:
            best_idx = train_tie_break[0]
        else:
            epoch_tie_break = np.max(train_tie_break)

            # 3rd tie break
            best_idx = epoch_tie_break
    
    f_exp.write('Best, %f, %f, %f\n' % (train_loss_list[best_idx],
                                        dev_score_list[best_idx],
                                        test_score_list[best_idx]))
    
    print 'Best'
    print 'train loss: ', train_loss_list[best_idx]
    print 'dev   loss: ', dev_score_list[best_idx]
    print 'test  loss: ', test_score_list[best_idx]

    return dev_answer_list[best_idx], test_answer_list[best_idx], train_loss_list[best_idx], dev_score_list[best_idx], test_score_list[best_idx]
    
def train_dataset(model, data):
    losses = []
    reg_losses = []
    avg_loss = 0.0
    avg_reg_loss = 0.0
    total_data = len(data) # number of training dataset
    for i, (tree, _) in enumerate(data): # (tree, label(dummy))
        reg_loss, loss, pred_y = model.train_step(tree, None)  # labels will be determined by model
        losses.append(loss)
        reg_losses.append(reg_loss)
        avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        avg_reg_loss = avg_reg_loss * (len(reg_losses) - 1) / len(reg_losses) + reg_loss / len(reg_losses)
        print 'avg loss %.5f at example %d of %d\r' % (avg_loss, i, total_data),
        # print 'avg reg loss %.2f at example %d of %d\r' % (avg_reg_loss, i, total_data),

    return np.mean(losses), np.mean(reg_losses)


def evaluate_dataset(model, data):
    num_correct = 0
    for tree, label in data:
        pred_y = model.predict(tree)[-1]  # root pred is final row
        num_correct += (label == np.argmax(pred_y))

    return float(num_correct) / len(data)

def read_exp1_dataset(dataset):
    data = read_discourse_dataset(dataset, DISCOURSE_ROLE, TYPE_GRANULARITY)
    return_data = {}
    if dataset == 'imdb' or dataset == 'tiny_dataset':
        return_data = data
    elif dataset == 'cornell' or dataset== 'sarcasm':
        return_data = read_ten_fold_dataset(data) # dev=8, test=9
    else:
        raise Exception('not supported dataset')
    return return_data

def read_exp4_dataset(dataset):
    data = read_discourse_dataset(dataset, DISCOURSE_ROLE, TYPE_GRANULARITY)
    return_data = {}
    if dataset == 'imdb' or dataset == 'tiny_dataset':
        return_data = data
    elif dataset == 'cornell' or dataset== 'sarcasm':
        return_data = read_ten_fold_dataset(data) # dev=8, test=9
    else:
        raise Exception('not supported dataset')
    return return_data


def read_ten_fold_dataset(pre_loaded_data, dev_fold=8, test_fold=9):
    return_data = {}
    train_folds = range(10)
    train_folds.remove(dev_fold)
    train_folds.remove(test_fold)
    data_train = []
    for i in train_folds:
        data_train = data_train + pre_loaded_data[str(i)]
    return_data['train'] = data_train
    return_data['dev'] = pre_loaded_data[str(dev_fold)]
    return_data['test'] = pre_loaded_data[str(test_fold)]
    return_data['max_degree'] = pre_loaded_data['max_degree']
    return return_data


def prepare_result_files():
    f_exp = open(HOME_PATH + 'result/discourse_exp'+EXP_VER+'_'+HOST_NAME+'.csv', 'a', 0)
    write_params_on_file(f_exp)
    f_answersheet_dev = open(HOME_PATH + 'result/discourse_exp'+EXP_VER+'_'+HOST_NAME+'_answersheet_dev.csv', 'a')
    write_params_on_file(f_answersheet_dev)
    f_answersheet_test = open(HOME_PATH + 'result/discourse_exp'+EXP_VER+'_'+HOST_NAME+'_answersheet_test.csv', 'a')
    write_params_on_file(f_answersheet_test)
    return f_exp, f_answersheet_dev, f_answersheet_test
    
def write_params_on_file(f):
    f.write('DATASET:%s\n' % DATASET)
    f.write('OPTIMIZER:%s\n' % OPTIMIZER)
    f.write('LEARNING_RATE:%.15f\n' % LEARNING_RATE)
    f.write('EMB_DIM:%d\n' % EMB_DIM)
    f.write('HIDDEN_DIM:%d\n' % HIDDEN_DIM)
    f.write('NUM_EPOCHS:%d\n' % NUM_EPOCHS)
    f.write('EPOCH_TRAIN_SHUFFLE:%d\n' % EPOCH_TRAIN_SHUFFLE)
    f.write('WHICH_CELL:%s\n' % WHICH_CELL)
    f.write('TYPE_GRANULARITY:%d\n' % TYPE_GRANULARITY)
    f.write('IS_N_ARY:%d\n' % IS_N_ARY)
    f.write('DISCOURSE_ROLE:%d\n' % DISCOURSE_ROLE)
    f.write('DISCOURSE_TYPE:%d\n' % DISCOURSE_TYPE)
    f.write('L2_REG_PARAM:%.15f\n' % L2_REG_PARAM)
    f.write('DROPOUT_RATIO:%.1f\n' % DROPOUT_RATIO)


def get_answer_each_epoch(model, data):
    return_list = []
    for tree, label in data:
        pred_y = model.predict(tree)[-1] # root pred is final row
        return_list.append(np.argmax(pred_y))
    return return_list

def write_answersheet(f_answersheet, data, answer):
    f_answersheet.write('file_name, pred, true\n')
    i=0
    for tree, label in data:
        pred = answer[i]
        i += 1
        f_answersheet.write('%s, %d, %d\n' % (tree.file_name, pred, label))

        
if __name__ == '__main__':
    train()
