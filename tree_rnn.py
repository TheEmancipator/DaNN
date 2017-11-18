#!/usr/bin/env python
# -*- coding: utf-8 -*-

# An implimentation of the Tree GRUs.
# Original Author: Ofir Nachum (ofirnachum@google.com)
# Customized by Brian Kangwook Lee (chaximeer@kaist.ac.kr) & Sanggyu Han (hsg1991@kaist.ac.kr)

__doc__ = """Implementation of Tree RNNs, and adaptation of RNNs to trees."""

import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict
from nn_inputs import gen_nn_inputs
from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams


theano.config.floatX = 'float32'


class TreeRNN(object):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """

    def __init__(self, num_emb, emb_dim, hidden_dim, output_dim,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=False,
                 optimizer=None,
                 discourse_type=False,
                 dropout_ratio=0.,
                 num_discourse_emb=35,
                 discourse_emb_dim=10,
                 trainable_type_embeddings=True):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        #TODO: For discourse embedding (now, hard coded)
        if discourse_type:
            self.num_discourse_emb = num_discourse_emb # Except 'root'
            self.discourse_emb_dim = discourse_emb_dim # Heuristically selected (Usually, <= 10)
            trainable_discourse_embeddings = trainable_type_embeddings

        #TODO: refactoring... (dropout)
        self.srng = RandomStreams(seed=7753)

        #TODO: refactoring... (adam)
        # For adam optimizerlr
        beta_1=0.9
        beta_2=0.999
        epsilon=1e-8
        decay=0.
        rho=0.9
        
        self.updates = []
        # self.weights = []
        self.iterations = K_variable(0.)
        self.lr = K_variable(learning_rate)
        self.beta_1 = K_variable(beta_1)
        self.beta_2 = K_variable(beta_2)
        self.epsilon = epsilon
        self.decay = K_variable(decay)
        self.rho = K_variable(rho)
        self.initial_decay = decay

        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim])) #TODO: exp3 - Regularization (embedding layer)
        if trainable_embeddings:
            self.params.append(self.embeddings)

        self.x = T.ivector(name='x')  # word indices
        self.tree = T.imatrix(name='tree')  # shape [None, self.degree]
        if labels_on_nonroot_nodes:
            self.y = T.fmatrix(name='y')  # output shape [None, self.output_dim]
            self.y_exists = T.fvector(name='y_exists')  # shape [None]
        else:
            self.y = T.fvector(name='y')  # output shape [self.output_dim]

        self.num_words = self.x.shape[0]  # total number of nodes (leaves + internal) in tree
        emb_x = self.embeddings[self.x]
        emb_x = emb_x * T.neq(self.x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings

        #TODO: For discourse embedding (now, hard coded)
        if discourse_type:
            self.discourse_embeddings = theano.shared(self.init_matrix([self.num_discourse_emb, self.discourse_emb_dim]))
            if trainable_discourse_embeddings:
                self.params.append(self.discourse_embeddings)
            self.discourse_x = T.ivector(name='discourse_x')
            discourse_emb_x = self.discourse_embeddings[self.discourse_x]
            discourse_emb_x = discourse_emb_x * T.neq(self.discourse_x, -1).dimshuffle(0, 'x') # dimshuffle: N to Nx1
            # emb_x = T.concatenate([emb_x, discourse_emb_x], axis=1)
            # self.emb_dim += self.discourse_emb_dim

        if dropout_ratio != 0:
            mask_emb_x = self.srng.binomial(size=emb_x.shape, p=1.-dropout_ratio) # p=1-dropout_ratio because 1's indicate keep and p is prob of dropping
            dropout_emb_x = emb_x * T.cast(mask_emb_x, theano.config.floatX)
            emb_x *= (1.- dropout_ratio)

            if discourse_type:
                mask_discourse_emb_x = self.srng.binomial(size=discourse_emb_x.shape, p=1.-dropout_ratio) # p=1-dropout_ratio because 1's indicate keep and p is prob of dropping
                dropout_discourse_emb_x = discourse_emb_x * T.cast(mask_discourse_emb_x, theano.config.floatX)
                discourse_emb_x *= (1.- dropout_ratio)


        if discourse_type:
            if dropout_ratio != 0:
                dropout_tree_states = self.compute_tree_disc_type(dropout_emb_x, dropout_discourse_emb_x, self.tree)
                mask_tree_states = self.srng.binomial(size=dropout_tree_states.shape, p=1-dropout_ratio)
                dropout_tree_states = dropout_tree_states * T.cast(mask_tree_states, theano.config.floatX)
                self.tree_states = self.compute_tree_disc_type(emb_x, discourse_emb_x, self.tree, is_first=False)
                self.tree_states *= (1.-dropout_ratio)
                dropout_tree_states = dropout_tree_states[:,:-self.discourse_emb_dim]
            else:
                self.tree_states = self.compute_tree_disc_type(emb_x, discourse_emb_x, self.tree) #TODO:bug#1

            self.tree_states = self.tree_states[:,:-self.discourse_emb_dim]
            self.final_state = self.tree_states[-1]

        else:
            if dropout_ratio != 0:
                dropout_tree_states = self.compute_tree(dropout_emb_x, self.tree)
                mask_tree_states = self.srng.binomial(size=dropout_tree_states.shape, p=1-dropout_ratio)
                dropout_tree_states = dropout_tree_states * T.cast(mask_tree_states, theano.config.floatX)
                self.tree_states = self.compute_tree(emb_x, self.tree, is_first=False)
                self.tree_states *= (1.-dropout_ratio)
            else:
                self.tree_states = self.compute_tree(emb_x, self.tree) #TODO:bug#1

            self.final_state = self.tree_states[-1]

        if labels_on_nonroot_nodes:
            self.output_fn = self.create_output_fn_multi() #TODO:bug#1
            self.pred_y = self.output_fn(self.tree_states)
            if dropout_ratio != 0:
                self.loss, self.origin_loss = self.loss_fn_multi(self.y, self.output_fn(dropout_tree_states), self.y_exists)
            else:
                self.loss, self.origin_loss = self.loss_fn_multi(self.y, self.pred_y, self.y_exists)

        else:
            self.output_fn = self.create_output_fn()
            self.pred_y = self.output_fn(self.final_state)
            if dropout_ratio != 0:
                self.loss = self.loss_fn(self.y, self.output_fn(dropout_tree_states))
            else:
                self.loss = self.loss_fn(self.y, self.pred_y)

        if optimizer == 's':
            updates = self.gradient_descent(self.loss)
        elif optimizer == 'a':
            updates = self.adam(self.loss)
        elif optimizer == 'r':
            # raise Exception('Not yet implemented!')
            updates = self.RMSprop(self.loss)
        else:
            raise Exception('optimizer has wrong value!')

        train_inputs = [self.x, self.tree, self.y]
        pred_inputs = [self.x, self.tree]
        if labels_on_nonroot_nodes:
            train_inputs.append(self.y_exists)
        if discourse_type:
            train_inputs.append(self.discourse_x)
            pred_inputs.append(self.discourse_x)

        self._train = theano.function(train_inputs,
                                      [self.loss, self.origin_loss, self.pred_y],
                                      updates=updates)

        self._evaluate = theano.function(pred_inputs,
                                         self.final_state)

        self._predict = theano.function(pred_inputs,
                                        self.pred_y)

    def _check_input(self, x, tree):

        # Special case - only root node exists
        if tree[0][0] == -1:
            assert len(x)==1

        # General case
        else:
            # print '_check_input():', tree[:,-1]
            # print '_check_input():', np.arange(len(x) - len(tree), len(x))
            assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))
            if not self.irregular_tree:
                assert np.all((tree[:, 0] + 1 >= np.arange(len(tree))) |
                              (tree[:, 0] == -1))
                assert np.all((tree[:, 1] + 1 >= np.arange(len(tree))) |
                              (tree[:, 1] == -1))

    def train_step_inner(self, x, tree, y):
        self._check_input(x, tree)
        return self._train(x, tree[:, :-1], y)

    def train_step(self, root_node, y):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        return self.train_step_inner(x, tree, y)

    def evaluate(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._evaluate(x, tree[:, :-1])

    def predict(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._predict(x, tree[:, :-1])

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out]) #TODO: exp3 - Regularization (output layer)

        def fn(final_state):
            return T.nnet.softmax(
                T.dot(self.W_out, final_state) + self.b_out)
        return fn

    def create_output_fn_multi(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out]) #TODO: exp3 - Regularization (output layer)

        def fn(tree_states):
            return T.nnet.softmax(
                T.dot(tree_states, self.W_out.T) +
                self.b_out.dimshuffle('x', 0))
        return fn

    # def create_recursive_unit(self):
    #     self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
    #     self.W_hh = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
    #     self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
    #     self.params.extend([self.W_hx, self.W_hh, self.b_h]) #TODO: exp3 - Regularization (recurrent layer)
    #     def unit(parent_x, child_h, child_exists):  # very simple
    #         h_tilde = T.sum(child_h, axis=0)
    #         h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
    #         return h
    #     return unit

    def create_final_unit(self):
        self.W_hxr = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.W_hhr = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_hr = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([self.W_hxr, self.W_hhr, self.b_hr])
        def unit(parent_x, child_h, child_exists):  # very simple
            h_tilde = T.sum(child_h, axis=0)
            h = T.tanh(self.b_hr + T.dot(self.W_hxr, parent_x) + T.dot(self.W_hhr, h_tilde))
            return h
        return unit

    def create_root_unit(self):
        dummy = 0 * theano.shared(self.init_matrix([self.degree, self.hidden_dim]))
        def unit(root_x):
            return self.final_unit(root_x, dummy, dummy.sum(axis=1))
        return unit

    def create_recursive_unit(self, is_first):
        if is_first:
            self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
            self.W_hh = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_hx, self.W_hh, self.b_h]) #TODO: exp3 - Regularization (recurrent layer)
        def unit(parent_x, child_h, child_exists):  # very simple
            h_tilde = T.sum(child_h, axis=0)
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
            return h
        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_matrix([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(leaf_x, dummy, dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree, is_first=True): #TODO:bug#1
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # Special case - only root node exists
        if tree[0][0] == -1:
            self.final_unit = self.create_final_unit()
            self.root_unit = self.create_root_unit()
            # compute only root hidden states
            root_h, _ = theano.map(
                fn=self.root_unit,
                sequences=[emb_x])
            return root_h
        #TODO: is emb_x matrix?

        # General case
        else:
            # compute leaf hidden states
            self.recursive_unit = self.create_recursive_unit(is_first)
            self.leaf_unit = self.create_leaf_unit()

            leaf_h, _ = theano.map(
                fn=self.leaf_unit,
                sequences=[emb_x[:num_leaves]])
            if self.irregular_tree:
                init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
            else:
                init_node_h = leaf_h

            # use recurrence to compute internal node hidden states
            def _recurrence(cur_emb, node_info, t, node_h, last_h):
                child_exists = node_info > -1
                offset = num_leaves * int(self.irregular_tree) - child_exists * t
                child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
                parent_h = self.recursive_unit(cur_emb, child_h, child_exists)
                node_h = T.concatenate([node_h,
                                        parent_h.reshape([1, self.hidden_dim])])
                return node_h[1:], parent_h

            dummy = theano.shared(self.init_vector([self.hidden_dim]))
            (_, parent_h), _ = theano.scan(
                fn=_recurrence,
                outputs_info=[init_node_h, dummy],
                sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
                n_steps=num_nodes)

            return T.concatenate([leaf_h, parent_h], axis=0)


    def create_recursive_unit_disc_type(self, is_first):
        if is_first:
            self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
            self.W_hh = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim+self.discourse_emb_dim]))
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_hx, self.W_hh, self.b_h]) #TODO: exp3 - Regularization (recurrent layer)
        def unit(parent_x, parent_discourse_x, child_h, child_exists):  # very simple
            h_tilde = T.sum(child_h, axis=0)
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
            h = T.concatenate([h, parent_discourse_x])
            return h
        return unit

    def create_leaf_unit_disc_type(self):
        dummy_child_h = 0 * theano.shared(self.init_matrix([self.degree, self.hidden_dim+self.discourse_emb_dim]))
        def unit(leaf_x, leaf_discourse_x):
            return self.recursive_unit(leaf_x, leaf_discourse_x, dummy_child_h, dummy_child_h.sum(axis=1))
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
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        else:
            init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, cur_discourse_emb, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h = self.recursive_unit(cur_emb, cur_discourse_emb, child_h, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim+self.discourse_emb_dim])])
            return node_h[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim+self.discourse_emb_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[emb_x[num_leaves:], discourse_emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)


    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)

    def gradient_descent(self, loss):
        """Momentum GD with gradient clipping."""
        grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates
    
    

    #TODO: refactoring... (adam)

    
    def adam(self, loss):    
        grads = T.grad(loss, self.params)
        self.updates = [(self.iterations, self.iterations + 1)]
        
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
        t = self.iterations + 1
        lr_t = lr * K_sqrt(1. - K_pow(self.beta_2, t)) / (1. - K_pow(self.beta_1, t))

        shapes = [K_get_variable_shape(p) for p in self.params]
        ms = [K_zeros(shape) for shape in shapes]
        vs = [K_zeros(shape) for shape in shapes]
        # self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(self.params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K_square(g)
            p_t = p - lr_t * m_t / (K_sqrt(v_t) + self.epsilon)

            self.updates.append(K_update(m, m_t))
            self.updates.append(K_update(v, v_t))

            new_p = p_t
            # apply constraints
            # if p in constraints:
                # c = constraints[p]
                # new_p = c(new_p)
            self.updates.append(K_update(p, new_p))
        return self.updates
    
    
    def RMSprop(self, loss):
        grads = T.grad(loss, self.params)
        shapes = [K_get_variable_shape(p) for p in self.params]
        accumulators = [K_zeros(shape) for shape in shapes]
        # self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K_update_add(self.iterations, 1))

        for p, g, a in zip(self.params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K_square(g)
            self.updates.append(K_update(a, new_a))
            new_p = p - lr * g / (K_sqrt(new_a) + self.epsilon)

            # apply constraints
            # if p in constraints:
            #     c = constraints[p]
            #     new_p = c(new_p)
            self.updates.append(K_update(p, new_p))
        return self.updates


class NaryTreeRNN(TreeRNN):

    def create_recursive_unit(self, is_first):
        if is_first:
            self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
            self.W_hh = theano.shared(self.init_matrix([self.degree, self.hidden_dim, self.hidden_dim]))
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_hx, self.W_hh, self.b_h])

        def unit(parent_x, child_h, child_exists):
            h_tilde, _ = theano.map(
                fn=lambda Whh, h, exists:
                    exists * T.dot(Whh, h),
                sequences=[self.W_hh, child_h, child_exists])
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + h_tilde.sum(axis=0))
            return h
        return unit

    def create_recursive_unit_disc_type(self, is_first):
        if is_first:
            self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
            self.W_hh = theano.shared(self.init_matrix([self.degree, self.hidden_dim, self.hidden_dim+self.discourse_emb_dim]))
            self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
            self.params.extend([self.W_hx, self.W_hh, self.b_h])

        def unit(parent_x, parent_discourse_x, child_h, child_exists):
            h_tilde, _ = theano.map(
                fn=lambda Whh, h, exists:
                    exists * T.dot(Whh, h),
                sequences=[self.W_hh, child_h, child_exists])
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + h_tilde.sum(axis=0))
            h = T.concatenate([h, parent_discourse_x])
            return h
        return unit

    
def K_update_add(x, increment):
    return (x, x + increment)

def K_sqrt(x):
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)

def K_pow(x, a):
    return T.pow(x, a)

def K_get_variable_shape(x):
    return x.get_value(borrow=True, return_internal_type=True).shape

def K_zeros(shape, dtype=theano.config.floatX):
    '''Instantiates an all-zeros variable.
    '''
    return K_variable(np.zeros(shape), dtype)

def K_variable(value, dtype=theano.config.floatX):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, strict=False)
    

def K_square(x):
    return T.sqr(x)

def K_update(x, new_x):
    return (x, new_x)