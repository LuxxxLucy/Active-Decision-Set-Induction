
import Orange
# import local package
from .BRS.model import BRS

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

    # import general package
    import numpy as np
    np.random.seed(random_seed)
    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(dataset.X)
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    # fit the explainer to the data
    target_class = dataset.domain.class_var.values[target_class_idx]
    explainer = BRS(dataset, blackbox,target_class_idx)
    rule_set = explainer.fit(dataset.domain,dataset.X,dataset.Y)

    # convert the rule representation
    rule_set = explainer.rules_convert(rule_set,dataset.domain)
    return rule_set

def compute_metrics_brs(rules):
    print("number of rules",len(rules))
    if len(rules) != 0:
        print("ave number of conditions" , sum([ len(r.conditions) for r in rules]) / len(rules) )
        print("max number of conditions" , max([ len(r.conditions) for r in rules]) )
        print("used features", len(set(  [ c.column for r in rules for c in r.conditions]  ) )  )
    else:
        print("ave number of conditions" , 0 )
        print("max number of conditions" , 0 )
        print("used features", 0 )
    return
