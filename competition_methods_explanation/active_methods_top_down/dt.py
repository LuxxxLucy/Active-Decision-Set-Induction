'''
    the algorithm is a modification of the Trepan algorithm as a Monte-Carlo tree-search problem
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
from .dtextract.examples.runCompare import *
from .dtextract.utils import rules_convert
# from utils import check

def explain_tabular_no_query(dataset, blackbox, target_class = 'yes', random_seed=42, K=-1,termination_max=20):
    '''
    this is a baseline where we don't actively query the black box to refine our pattern but only induce pattern from the labelled dataset.
    In fact, this is the same with pattern induction a dataset.
    '''
    return explain_tabular(dataset,blackbox, target_class = 'yes', random_seed=42, K=-1, active=False)

def explain_tabular(dataset, blackbox, target_class = 'yes', random_seed=42, K=-1, active=True,termination_max=20):
    '''
    `active` indicates
    '''

    # set random seed
    np.random.seed(random_seed)

    isClassify = True
    nComponents = 10


    XTrain = dataset.X
    yTrain = dataset.Y
    catFeatIndsTrain = [ i for i,e in enumerate(dataset.domain.attributes) if e.is_discrete]
    numericFeatIndsTrain = [ i for i,e in enumerate(dataset.domain.attributes) if e.is_continuous]


    paramsLearn = ParamsLearn(tgtScore, minGain, maxSize)
    paramsSimp = ParamsSimp(nPts, nTestPts, isClassify)

    dist = CategoricalGaussianMixtureDist(XTrain, catFeatIndsTrain, numericFeatIndsTrain, nComponents)

    rfFunc = blackbox

    # from simp import InternalNode
    for feature in dataset.domain.attributes:
        if feature.is_continuous:
            feature.max = max([ d[feature] for d in dataset ]).value
            feature.min = min([ d[feature] for d in dataset ]).value

    constraints = []
    for idx,a in enumerate(dataset.domain.attributes):
        if a.is_continuous:
            constraints.append( (InternalNode(idx,a.max),True) )
            constraints.append( (InternalNode(idx,a.min),False) )

    # constraints = [ (InternalNode(0,0.95),True ), (InternalNode(0,0),False ),(InternalNode(1,1.95), True ), (InternalNode(1,0), False )]

    # dtExtract = learnDTSimp(genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp)
    # dtExtract = learnDTSimp(genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp)
    dtExtract = learnDT(lambda rfFunc, dist, cons: genAxisAligned(rfFunc, dist, constraints, paramsSimp), rfFunc, dist, constraints, paramsLearn)

    rules = rules_convert(dtExtract,dataset,target_class=target_class)

    return rules
