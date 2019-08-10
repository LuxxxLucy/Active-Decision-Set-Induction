'''
    Input params:
    1. dataset. Note that there are mainly two parts in it
        * `dataset.X` and `dataset.Y`: the raw real instances
        * `dataset.domain`: which defines the domain of input space.
    2. blackbox: A blackbox outputing binary classification labels. Should follow a scikit API.
    3. target_class_idx: the value of the class we are interested in
    4. random_seed: for reproduction.
    5. K: the number of rules to return, return all rules if not specified
'''

import numpy as np
# from .mcts_core.tree import Tree
from .dtextract_cart.active_cart import Active_CART as Tree
# from .mcts_core.core import diverse_filter_sorted
# from utils import check

def explain_tabular_no_query(dataset, blackbox, target_class = 'yes', random_seed=42, K=-1,termination_max=20):
    '''
    this is a baseline where we don't actively query the black box to refine our pattern but only induce pattern from the labelled dataset.
    In fact, this is the same with pattern induction a dataset.
    '''
    return explain_tabular(dataset,blackbox, target_class = 'yes', random_seed=42,active_instance_number=0, K=-1, active=False)

def explain_tabular(dataset, blackbox, target_class_idx = 1, random_seed=42, active_instance_number=200, active=True,termination_max=20):
    '''
    `active` indicates
    '''

    # set random seed
    np.random.seed(random_seed)


    tree = Tree(dataset,dataset.domain,target_class_idx,blackbox=blackbox,active_instance_number=active_instance_number,max_iteration=termination_max)
    tree.fit(dataset.X,dataset.Y)
    # from .dtextract_cart.cart import CART
    # tree = CART(max_depth=100)
    # tree.fit(dataset.X,dataset.Y)
    # tree.fit(data)

    # tree.print_tree()
    explanations = tree.output_all()
    # explanations = None

    return explanations,tree

def compute_metrics_dt(rules):
    '''compute various metrics for rules'''

    print("number of rules",len(rules))
    print("ave number of conditions" , sum([ len(r.conditions) for r in rules]) / len(rules) )
    print("max number of conditions" , max([ len(r.conditions) for r in rules]) )
    print("used features", len(set(  [ c.column for r in rules for c in r.conditions]  ) )  )
