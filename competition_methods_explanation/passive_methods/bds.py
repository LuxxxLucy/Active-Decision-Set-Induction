# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from .BDS.model import BDS

def explain_tabular(dataset, blackbox, target_class='yes', pre_label=True, random_seed=42):
    '''
    Input Params:
    1. dataset: a Orange data table
    3. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    4. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)

    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(dataset.X)
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    # fit the explainer to the data
    explainer = BDS(dataset, blackbox)
    print("pre generation okay, quit")
    rule_set = explainer.fit(dataset.domain,dataset.X,dataset.Y,target_class=target_class)

    # convert the rule representation
    rule_set = explainer.rules_convert(rule_set,dataset.domain)
    return rule_set
