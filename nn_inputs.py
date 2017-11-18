
import numpy as np
import theano
theano.config.floatX = 'float32'

class Node(object):
    def __init__(self, val=None):
        self.children = []
        self.val = val
        self.idx = None
        self.height = 1
        self.size = 1
        self.num_leaves = 1
        self.parent = None
        self.label = None
        self.file_name = None
        self.discourse_relation_type = None # my discourse relation type
        self.discourse_role = None # children's discourse role - for the sake of convenience

    def _update(self):
        self.height = 1 + max([child.height for child in self.children if child] or [0])
        self.size = 1 + sum(child.size for child in self.children if child)
        self.num_leaves = (all(child is None for child in self.children) +
                           sum(child.num_leaves for child in self.children if child))
        if self.parent is not None:
            self.parent._update()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update()

    def add_children(self, other_children):
        self.children.extend(other_children)
        for child in other_children:
            child.parent = self
        self._update()


def gen_nn_inputs_disc_type(root_node,
                            max_degree            =None,
                            only_leaves_have_vals =True,
                            with_labels           =True):
    _clear_indices(root_node)
    x, leaf_labels, leaf_disc_types = _get_leaf_vals_disc_type(root_node)
    tree, internal_x, internal_labels, internal_disc_types = \
        _get_tree_traversal_disc_type(root_node, len(x), max_degree)
    assert all(v is not None for v in x)
    if only_leaves_have_vals:
        assert all(v is -1 for v in internal_x)
    x.extend(internal_x)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    # with_labels: include labels/labels_exists on return value
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        disc_types = leaf_disc_types + internal_disc_types
        # labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'), #TODO:bug#1
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX),
                np.array(disc_types, dtype='int32'))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


# Get leaf values in deep-to-shallow, left-to-right order.
def _get_leaf_vals_disc_type(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children): # if children is empty
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer
    vals = []
    labels = []
    disc_types = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
        labels.append(leaf.label)
        disc_types.append(leaf.discourse_relation_type)
    return vals, labels, disc_types


def _get_tree_traversal_disc_type(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_vals = []
    labels = []
    disc_types = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_vals.append(node.val if node.val is not None else -1)
            labels.append(node.label)
            disc_types.append(node.discourse_relation_type)
            idx += 1

    return tree, internal_vals, labels, disc_types


def gen_nn_inputs(root_node,
                  max_degree            =None,
                  only_leaves_have_vals =True,
                  with_labels           =False):

    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x   : the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)

    x, leaf_labels = _get_leaf_vals(root_node)
    tree, internal_x, internal_labels = \
        _get_tree_traversal(root_node, len(x), max_degree)
    assert all(v is not None for v in x)
    #TODO: integrity check w/ only_leaves_have_vals flag
    if only_leaves_have_vals:
        assert all(v is -1 for v in internal_x)
    x.extend(internal_x)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)

    # with_labels: include labels/labels_exists on return value
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        # labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'), #TODO:bug#1
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


# Get leaf values in deep-to-shallow, left-to-right order.
def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children): # if children is empty
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    vals = []
    labels = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
        labels.append(leaf.label)
    return vals, labels # vals & labels of leaves


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_vals = []
    labels = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_vals.append(node.val if node.val is not None else -1)
            labels.append(node.label)
            idx += 1

    return tree, internal_vals, labels
