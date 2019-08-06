# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from pysbrl import BayesianRuleList

def explain_tabular(dataset, blackbox, target_class_idx=1, pre_label=True, random_seed=42):
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
    target_class = dataset.domain.class_var.values[target_class_idx]

    print("first discretize data since SBRL only handles categorical data")
    # tmp_table = Orange.data.Table.from_numpy(X=dataset.X,Y=dataset.Y,domain=dataset.domain)
    # disc = Orange.preprocess.Discretize()
    # disc.method = Orange.preprocess.discretize.EntropyMDL()
    # disc_dataset = disc(tmp_table)
    disc_dataset = dataset
    explainer = BayesianRuleList()
    # explainer = BayesianRuleList(lambda_=100)
    # explainer = BayesianRuleList(lambda_=40,eta=4)
    print("initialize okay")
    print("start finding rule list")
    rule_list = explainer.fit((disc_dataset.X).astype(int),(disc_dataset.Y==target_class_idx).astype(int) )
    print("finished finding rule list")

    # convert the rule representation
    # rule_list = explainer.rules_convert(rule_list,dataset.domain)
    return explainer

def compute_metrics_sbrl(sbrl_rule_list):
    print("number of rules",len(sbrl_rule_list))
    print("ave number of conditions" , sum([ len(r[0]) for r in sbrl_rule_list]) / len(sbrl_rule_list) )
    print("max number of conditions" , max([ len(r[0]) for r in sbrl_rule_list]) )
    print("used features", len(set(  [ c.feature_idx for r in sbrl_rule_list for c in r[0]]  ) )  )
    return
