# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange
from Orange.classification.rules import _RuleLearner,Rule,RuleHunter
from Orange.classification.rules import LaplaceAccuracyEvaluator,WeightedRelativeAccuracyEvaluator

# import local package
from .anchor import anchor_tabular
# from .anchor.anchor_explanation import AnchorExplanation

# two functions are of interest:
# 1. `cn2_tabular`: explain though the CN2 algorithm (only for one class)
# 2. `cn2sd_tabular`: explain though the CN2SD algorithm (only for one class)
# ---
# Input Params:
# 1. dataset: a Orange data table
# 2. encoder: a scikit transformer
# 3. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
# 4. target_class
class Selector(namedtuple('Selector', 'column, op, value')):

    OPERATORS = {
        # discrete, nominal variables
        '==': operator.eq,
        '=': operator.eq,
        '!=': operator.ne,
        # continuous variables
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
    }
    def filter_instance(self, x):
        """
        Filter a single instance. Returns true or false
        """
        return Selector.OPERATORS[self[1]](x[self[0]], self[2])

    def filter_data(self, X):
        """
        Filter several instances concurrently. Retunrs array of bools
        """
        return Selector.OPERATORS[self[1]](X[:, self[0]], self[2])

def cn2anchor_tabular(dataset, blackbox, target_class = 'yes', random_seed=42):
    '''
    this is a warpper of the cn2 algorithm where the rule finder is replaced by Anchor algorithm
    The only changes are:
    Note only rules for one class (by default class 1) is generated, while the original algorithm will consider every class.
    '''
    np.random.seed(random_seed)

    # re-labelled the data
    labels = blackbox(dataset.X)
    data_table = Orange.data.Table(dataset.domain, dataset.X, labels)


    class AnchorRuleFinder(RuleHunter):
        '''
        Rule Hunter is the Rule Finder algorithm implemented in the Orange package.
        To use Anchor, we only needs to warp the __call__ function
        '''
        def __init__(self,data_table,blackbox):
            super().__init__()

            class_names = data_table.domain.class_var.values
            feature_names = [ it.name for it in data_table.domain.attributes ]
            data = data_table.X
            categorical_names = {  i: it.values for i,it in enumerate(data_table.domain.attributes) if it.is_discrete }
            self.anchor_explainer = anchor_tabular.AnchorTabularExplainer(class_names, feature_names, data, encoder=None, discretizer=None, categorical_names=categorical_names)
            self.blackbox = blackbox

        def __call__(self, X, Y, W, target_class, domain):
            '''
            the warping is relatively simple, we
            '''
            self.anchor_explainer.fit(X,Y,X,Y)

            # randomly pick a instance of that target_class
            idx = random.choice( np.where(Y==target_class)[0] )

            anchor_exp = self.anchor_explainer.explain_instance(X[idx], blackbox, threshold=0.99)
            # note that anchor exp is a different data structure
            # a converting is needed
            string_exp = anchor_exp.exp_map['names']
            # translating to string representation
            conditions = [ t.split(' ') for t in string_exp]
            selectors=[]
            col_names = [it.name for it in domain.attributes]
            for condition in conditions:
                # this is tricky because some anchor rule is like "1 < X < 5".
                # and should be translated into two rules : 1. "X > 1" and 2. "X < 5"
                if len(condition)==3:
                    col_idx = col_names.index(condition[0])
                    if domain.attributes[col_idx].is_continuous:
                        value = float(condition[2])
                    s = Selector(column=col_idx,op=condition[1],value=value)
                    selectors.append(s)
                elif len(condition) == 5:
                    col_idx = col_names.index(condition[2])
                    if domain.attributes[col_idx].is_continuous:
                        condition[4] = float(condition[4])
                        condition[0] = float(condition[0])
                    s = Selector(column=col_idx,op=condition[3],value=condition[4])
                    selectors.append(s)
                    reverser_op = lambda x: {'<':'>','<=':'>=','=':'=','==':'==','>':'<','>=':'<='}[x]
                    s = Selector(column=col_idx,op= reverser_op(condition[1]),value=condition[0])
                    selectors.append(s)

            rule = Rule(selectors=selectors,domain=domain)
            rule.prediction = target_class

            # compute the covered data
            # note covered instance of target class is removed
            rule.covered_examples = np.ones(X.shape[0], dtype=bool)
            for selector in rule.selectors:
                rule.covered_examples &= selector.filter_data(X)
            print(string_exp)
            print(np.sum(rule.covered_examples))
            rule.covered_examples &= Y==1
            print(np.sum(rule.covered_examples))
            return rule

    #intializes the explainer
    class cn2anchor_explainer(_RuleLearner):
        def __init__(self, data_table,blackbox,preprocessors=None, base_rules=None):
            super().__init__(preprocessors, base_rules)
            # encoder is a now only a decorator.
            self.rule_finder = AnchorRuleFinder(data_table,blackbox)
            self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

        def find_rules(self, X, Y, W, target_class, domain):
            rule_list = []
            while not self.data_stopping(X, Y, W, target_class):
                # generate a new rule that has not been seen before
                new_rule = self.rule_finder(X, Y, W, target_class, domain )

                # the general requirements can be found
                if len(rule_list) >= 20:
                    print("max number rule generated. exiting now")
                    break
                # if new_rule == None:
                #     print("None Rule Generated")
                #     break
                # exclusive removed the covered instance of target class
                X, Y, W = self.cover_and_remove(X, Y, W, new_rule)
                rule_list.append(new_rule)

            return rule_list

        def fit(self, X, Y, domain,target_class,W=None):
            self.domain = domain
            Y = Y.astype(dtype=int)
            rule_list = []

            target_class_idx = domain.class_var.values.index(target_class)

            # rule_list.extend(self.find_rules(X, Y, W, target_class_idx,self.domain))
            rule_list = self.find_rules(X, Y, W, target_class_idx,self.domain)
            # add the default rule
            # rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
            return rule_list



    # fit the explainer to the data
    explainer = cn2anchor_explainer(data_table,blackbox)
    rule_list = explainer.fit(data_table.X,data_table.Y,data_table.domain,target_class=target_class)

    return rule_list

def cn2anchorsd_tabular(dataset,encoder, blackbox, target_class = 'yes', random_seed=42):
    '''
    this is a warpper of the cn2 algorithm where the rule finder is replaced by Anchor algorithm
    The only changes are:
    1. the labels are re-labeled by the blackbox
    2. only rules for one class (by default class 1) is generated, while the original algorithm will consider every class.
    '''
    pass
    # np.random.seed(random_seed)
    #
    # # re-labelled the data
    # labels = blackbox(encoder(dataset.X))
    # data_table = Orange.data.Table(dataset.domain, dataset.X, labels)
    #
    #
    # #intializes the explainer
    # class cn2sd_explainer(_RuleLearner):
    #     def __init__(self, preprocessors=None, base_rules=None):
    #         super().__init__(preprocessors, base_rules)
    #         self.rule_finder.quality_evaluator = WeightedRelativeAccuracyEvaluator()
    #         self.cover_and_remove = self.weighted_cover_and_remove
    #         # self.gamma = 0.7
    #         self.gamma = 0.2
    #
    #     def fit(self, X, Y, domain,target_class,W=None):
    #         self.domain = domain
    #         Y = Y.astype(dtype=int)
    #         rule_list = []
    #
    #         target_class_idx = domain.class_var.values.index(target_class)
    #
    #         rule_list.extend(self.find_rules(
    #             X, Y, np.copy(W) if W is not None else None,
    #             target_class_idx, self.base_rules, self.domain))
    #         # add the default rule
    #         rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
    #         return rule_list
    #
    #
    # # fit the explainer to the data
    # explainer = cn2sd_explainer()
    # rule_list = explainer.fit(data_table.X,data_table.Y,data_table.domain,target_class=target_class)
    #
    # return rule_list
