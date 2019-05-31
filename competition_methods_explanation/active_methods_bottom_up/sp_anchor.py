from __future__ import print_function
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
from .anchor import anchor_tabular
from .anchor import utils

from .explanation import *
from .utils import discretizer_from_dataset

def submodular_pick_anchor(dataset, encoder, blackbox, precompute_explanation=True,file_name=None,random_seed=42,k=5):
    np.random.seed(random_seed)

    if not precompute_explanation:


        discretizer = discretizer_from_dataset(dataset)

        # initialize the explainer
        explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data, encoder, discretizer, dataset.categorical_names)
        # figuring out the data distribution, etc
        explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)
        # get explanation for each explanation

        from tqdm import tqdm
        # explanations = [ explainer.explain_instance(dataset.train[idx], blackbox, threshold=0.95) for idx in tqdm( range(len(dataset.train)) ) ]
        explanations = [ explainer.explain_instance(dataset.train[idx], blackbox, threshold=0.95) for idx in tqdm( range(400) )]
        # explanations = [ explainer.explain_instance(dataset.train[idx], blackbox, threshold=0.95) for idx in range(40)]

        import pickle

        pickle.dump( explanations, open( file_name, "wb" ) )

        return
    else:
        import pickle
        explanations =pickle.load( open( file_name, "rb" ) )

    covered = {}
    data= dataset.train
    for i, e in enumerate(explanations):
        samples = e.exp_map['examples'][-1]["covered"]
        idxs= np.where((data==np.array(samples)[:,None]).all(-1))[1]
        covered[i] = set(idxs)

    chosen_explanation_id = []
    all_covered = set()
    for i in range(k):
        best = (-1, -1)
        for j in covered:
            gain = len(all_covered.union(covered[j]))
            if gain > best[1]:
                best = (j, gain)
        all_covered = all_covered.union(covered[best[0]])
        chosen_explanation_id.append(best[0])
    chosen_explanations = [  explanations[i] for i in chosen_explanation_id ]
    return chosen_explanations

def explain_one_instance_anchor(idx, dataset, encoder, blackbox,random_seed=42):
    '''
    this function is purely making a call to the original Anchor implementation
    '''

    np.random.seed(random_seed)

    discretizer = discretizer_from_dataset(dataset)

    # initialize the explainer
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data, encoder, discretizer, dataset.categorical_names)
    # figuring out the data distribution, etc
    # explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)
    explainer.fit(dataset.train, dataset.labels_train, dataset.train, dataset.labels_train)

    # exp = explainer.explain_instance(dataset.data[idx], blackbox, threshold=0.95)
    result = explainer.explain_instance(dataset.train[idx], blackbox, threshold=0.95)
    explanation = Explanation(result.type,result.exp_map)
    return explanation
