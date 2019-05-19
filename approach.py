'''
    the algorithm of our approach
    the principle of design is that, code in this file should be pseudo-code like.
    where each essential function are implemented in `core.py`
    ---
    Input params:
    1. dataset. Note that there are mainly two parts in it
        * `dataset.X` and `dataset.Y`: the raw real instances
        * `dataset.domain`: which defines the domain of input space.
    2. blackbox: A blackbox outputing binary classification labels. Should follow a scikit API.
    3. encoder: encoding raw data to formats that blackbox take, for example: one-hot coding for categorical features or scaling normalization for continuous features. WHICH WE DO NOT CARE.
    4. target_class: the value of the class we are interested in
    5. random_seed: for reproduction.
    6. K: the number of rules to return, return all rules if not specified
'''

import numpy as np
from mcts import Tree
from core import diverse_filter_sorted
# from utils import check

def explain_tabular_no_query(dataset,encoder, blackbox, target_class = 'yes', random_seed=42, K=-1,termination_max=20):
    '''
    this is a baseline where we don't actively query the black box to refine our pattern but only induce pattern from the labelled dataset.
    In fact, this is the same with pattern induction a dataset.
    '''
    return explain_tabular(dataset,encoder, blackbox, target_class = 'yes', random_seed=42, K=-1, active=False)

def explain_tabular(dataset,encoder, blackbox, target_class = 'yes', random_seed=42, K=-1, active=True,termination_max=20):
    '''
    `active` indicates
    '''

    # set random seed
    np.random.seed(random_seed)

    # check dataset. dealing with missing value, etc.
    # and using blackbox to label these real instances if a label is not provided in advance.
    # also compute the max and min for each feature
    # TODO: check(dataset,encoder,blackbox)

    # If possible, re-use previous tree
    # tree = restore_tree()

    # MCTS search
    tree = Tree(dataset,target_class,blackbox=blackbox,encoder=encoder,active=active)
    # let termination be a simple counter
    termination = 0
    while(True):
        # traverse the tree based on exploration-exploitation criteria
        # find the node with highest value
        node_selected = tree.select_best_node()
        if node_selected == None:
            # it means all space is searched over, which might be happen in real world
            print("all space searched over")
            print("current termination count:",termination)
            break

        # decide whether split or not: expand the node if split and
        if tree.split_or_not(node_selected) == True:
            tree.expand(node=node_selected)
            # roll out estimate of new nodes
        else:
            print('not supposed to be here for now')
            tree.roll_out_estimate(node_selected)
        # updating is called in the roll_out_estimate and expand function
        # tree.updating()

        if termination>=termination_max:
            print("termination met!")
            break
        termination+=1

    # Diverse Selection
    nodes = tree.output_all()
    nodes = diverse_filter_sorted(nodes,tree,dataset)
    patterns = [node.pattern for node in nodes]

    # return a pre-defined number of ruls or all the rules (if not specified)
    return (patterns[:K],tree) if K != -1 else (patterns,tree)
