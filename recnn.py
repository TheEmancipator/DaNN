#!/usr/bin/env python
# -*- coding: utf-8 -*-

# An implimentation of the Recursive Neural Network (RecNN).
# Author: Brian Kangwook Lee (chaximeer@kaist.ac.kr) & Sanggyu Han (hsg1991@kaist.ac.kr)

__doc__ = """Implementation of Recursive Neural Network"""

import tree_rnn

import theano
from theano import tensor as T


class RecNN(tree_rnn.TreeRNN):
    def create_recursive_unit(self, is_first):
        if is_first:
            self.W_h = theano.shared(self.init_matrix([self.hidden_dim, 2*self.emb_dim])) # emb=200, hidden=200
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_h, self.b_h])
        def unit(child_h):  # very simple
            h_tilde = T.flatten(child_h)
            # h_tilde = T.sum(child_h, axis=0)
            # print 'fuck', h_tilde
            # h_tilde = child_h.reshape([2*self.emb_dim,1])
            h = T.tanh(self.b_h + T.dot(self.W_h, h_tilde))
            return h
        return unit

    def create_leaf_unit(self):
        def unit(leaf_x):
            return leaf_x
        return unit

    def compute_tree(self, emb_x, tree, is_first=True):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        self.recursive_unit = self.create_recursive_unit(is_first)
        self.leaf_unit = self.create_leaf_unit()

        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(node_info, t, node_h, last_h):
            child_exists = node_info > -1
            # non irregular tree = every node has children
            offset = num_leaves * int(self.irregular_tree) - child_exists * t # t = index of the current internal node
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x') # [2 x embedding_dim]
            parent_h = self.recursive_unit(child_h) # [embedding_dim x 1]
            #node_printed = theano.printing.Print('node value')(node_h)
            #parent_printed = theano.printing.Print('parent value')(parent_h)
            node_h = T.concatenate([node_h, parent_h.reshape([1, self.hidden_dim])], axis=0)
            #node_h = T.concatenate([node_printed, parent_printed.reshape([1, self.hidden_dim])], axis=0)
            #node_h = T.concatenate([node_printed, parent_printed.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)


    def create_recursive_unit_disc_type(self, is_first):
        if is_first:
            self.W_h = theano.shared(self.init_matrix([self.hidden_dim, 2*(self.emb_dim+self.discourse_emb_dim)])) # emb=200, hidden=200
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_h, self.b_h])
        def unit(parent_discourse_x, child_h):
            h_tilde = T.flatten(child_h)
            h = T.tanh(self.b_h + T.dot(self.W_h, h_tilde))
            h = T.concatenate([h, parent_discourse_x])
            return h
        return unit

    def create_leaf_unit_disc_type(self):
        def unit(leaf_x, leaf_discourse_x):
            h = T.concatenate([leaf_x, leaf_discourse_x])
            return h
        return unit

    def compute_tree_disc_type(self, emb_x, discourse_emb_x, tree, is_first=True):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        self.recursive_unit = self.create_recursive_unit_disc_type(is_first)
        self.leaf_unit = self.create_leaf_unit_disc_type()

        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves], discourse_emb_x[:num_leaves]])
        init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_discourse_emb, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h = self.recursive_unit(cur_discourse_emb, child_h)
            node_h = T.concatenate([node_h, parent_h.reshape([1, self.hidden_dim+self.discourse_emb_dim])], axis=0)
            return node_h[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim+self.discourse_emb_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[discourse_emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)