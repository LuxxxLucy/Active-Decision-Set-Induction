import Orange
from Orange.classification.rules import _RuleLearner,LaplaceAccuracyEvaluator,WeightedRelativeAccuracyEvaluator
import numpy as np

# two functions are of interest:
# 1. `cn2_tabular`: explain though the CN2 algorithm (only for one class)
# 2. `cn2sd_tabular`: explain though the CN2SD algorithm (only for one class)
# ---
# Input Params:
# 1. dataset: a Orange data table
# 3. blackbx: a blackbox predict function, such as `c.predict` where c is a scikit-classifier (including the encoder.transform function!)
# 4. target_class

def cn2_tabular(dataset,blackbox, target_class = 'yes', random_seed=42):
    '''
    this is a warpper of the cn2 algorithm.
    The only changes are:
    1. the labels are re-labeled by the blackbox
    2. only rules for one class (by default class 1) is generated, while the original algorithm will consider every class.
    '''
    np.random.seed(random_seed)

    # re-labelled the data
    labels = blackbox(dataset.X)
    data_table = Orange.data.Table(dataset.domain, dataset.X, labels)


    #intializes the explainer
    class cn2_explainer(_RuleLearner):
        def __init__(self, preprocessors=None, base_rules=None):
            super().__init__(preprocessors, base_rules)
            self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

        def fit(self, X, Y, domain,target_class,W=None):
            self.domain = domain
            Y = Y.astype(dtype=int)
            rule_list = []

            target_class_idx = domain.class_var.values.index(target_class)

            rule_list.extend(self.find_rules(X, Y, W, target_class_idx,
                                             self.base_rules, self.domain))
            # add the default rule
            rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
            return rule_list

    # fit the explainer to the data
    explainer = cn2_explainer()
    rule_list = explainer.fit(data_table.X,data_table.Y,data_table.domain,target_class=target_class)

    return rule_list

def cn2sd_tabular(dataset,blackbox, target_class = 'yes', random_seed=42):
    '''
    this is a warpper of the cn2 algorithm.
    The only changes are:
    1. the labels are re-labeled by the blackbox
    2. only rules for one class (by default class 1) is generated, while the original algorithm will consider every class.
    '''
    np.random.seed(random_seed)

    # re-labelled the data
    labels = blackbox(dataset.X)
    data_table = Orange.data.Table(dataset.domain, dataset.X, labels)


    #intializes the explainer
    class cn2sd_explainer(_RuleLearner):
        def __init__(self, preprocessors=None, base_rules=None):
            super().__init__(preprocessors, base_rules)
            self.rule_finder.quality_evaluator = WeightedRelativeAccuracyEvaluator()
            self.cover_and_remove = self.weighted_cover_and_remove
            # self.gamma = 0.7
            self.gamma = 0.2

        def fit(self, X, Y, domain,target_class,W=None):
            self.domain = domain
            Y = Y.astype(dtype=int)
            rule_list = []

            target_class_idx = domain.class_var.values.index(target_class)

            rule_list.extend(self.find_rules(
                X, Y, np.copy(W) if W is not None else None,
                target_class_idx, self.base_rules, self.domain))
            # add the default rule
            rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
            return rule_list


    # fit the explainer to the data
    explainer = cn2sd_explainer()
    rule_list = explainer.fit(data_table.X,data_table.Y,data_table.domain,target_class=target_class)

    return rule_list
