# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from model import ADS

def explain_tabular(dataset,encoder, blackbox, target_class='yes', pre_label=True, random_seed=42):
    '''
    Input Params:
    1. dataset: a Orange data table
    2. encoder: a scikit transformer
    3. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    4. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)

    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(encoder(dataset.X))
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    explainer = ADS(dataset, blackbox, encoder, target_class=target_class) # fit the explainer to the data
    explainer.set_parameters() # set hyperparameter
    explainer.generate_rule_space() # pre-processing: rule space generation
    explainer.initialize() # initialize
    while(explainer.termination_condition() == False):
        # the local search: main algorithm
        actions = explainer.generate_action_space() # generating possible actions
        explainer.act(actions) # select an action!
        explainer.update()


    # post-processing the rule representation
    rule_set = explainer.output()
    print("okay!")
    return rule_set
