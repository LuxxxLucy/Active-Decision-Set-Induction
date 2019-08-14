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
from .utils import Condition
from .utils import Rule as Our_Rule
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

def cn2anchor_tabular(dataset, blackbox, target_class_idx = 1, random_seed=42,number = 100):
    '''
    this is a warpper of the cn2 algorithm where the rule finder is replaced by Anchor algorithm
    The only changes are:
    Note only rules for one class (by default class 1) is generated, while the original algorithm will consider every class.
    '''
    np.random.seed(random_seed)
    target_class = dataset.domain.class_var.values[target_class_idx]


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
            threshold = 0.99
            rule_candidates = []
            for _ in range(1000):
                idx = random.choice( np.where(Y==target_class)[0] )
                rule,precision = self.find_anchor(X,Y,W,target_class,domain,idx,threshold)
                if precision <= threshold:
                    rule_candidates.append( (rule,precision))
                    continue
                else:
                    # print("find one!")
                    return rule

            print("find best one")
            return max(rule_candidates,key= lambda x: x[1] )[0]
            # return rule_candidates[0]

        def find_anchor(self,X,Y,W,target_class,domain,idx,threshold=0.95):

            anchor_exp = self.anchor_explainer.explain_instance(X[idx], blackbox, threshold=threshold)
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
                        if condition[1] in ['<','<=']:
                            s = Condition(column=col_idx,type="continuous",max_value=value,min_value=domain.attributes[col_idx].min)
                        elif condition[1] in ['>','>=']:
                            s = Condition(column=col_idx,type="continuous",min_value=value,max_value=domain.attributes[col_idx].max)
                        else:
                            print("converting error")
                    else:
                        assert condition[1] in ["=","=="],"converting error, in categorical"
                        value = domain.attributes[col_idx].values.index(condition[2])
                        s = Condition(column=col_idx,type="categorical",values=[value])
                    selectors.append(s)
                elif len(condition) == 5:
                    col_idx = col_names.index(condition[2])
                    if domain.attributes[col_idx].is_continuous:
                        condition[4] = float(condition[4])
                        condition[0] = float(condition[0])
                    # s = Condition(column=col_idx,op=condition[3],value=condition[4])
                    # selectors.append(s)
                    # reverser_op = lambda x: {'<':'>','<=':'>=','=':'=','==':'==','>':'<','>=':'<='}[x]
                    # s = Condition(column=col_idx,op= reverser_op(condition[1]),value=condition[0])
                    # selectors.append(s)
                    s = Condition(column=col_idx,type="continuous",min_value=condition[0],max_value=condition[4])

            rule = Our_Rule(conditions=selectors,target_class_idx=target_class)
            rule.target_class = domain.class_var.values[target_class]

            # compute the covered data
            # note covered instance of target class is removed
            rule.covered_examples = np.ones(X.shape[0], dtype=bool)
            for selector in rule.conditions:
                rule.covered_examples &= selector.filter_data(X)
            # print(string_exp)
            # print(np.sum(rule.covered_examples))
            rule.covered_examples &= Y==1
            # print(np.sum(rule.covered_examples))
            return rule,anchor_exp.precision()

    #intializes the explainer
    class cn2anchor_explainer(_RuleLearner):
        def __init__(self, data_table,blackbox,preprocessors=None, base_rules=None):
            super().__init__(preprocessors, base_rules)
            # encoder is a now only a decorator.
            self.rule_finder = AnchorRuleFinder(data_table,blackbox)
            self.rule_finder.quality_evaluator = LaplaceAccuracyEvaluator()

        def find_rules(self, X, Y, W, target_class, domain,number=10):
            rule_list = []

            max_iteration = number
            verbose = True
            if verbose:
                from tqdm import tqdm_notebook as tqdm
                pbar = tqdm(total=max_iteration)
            while not self.data_stopping(X, Y, W, target_class,domain=domain):
                # generate a new rule that has not been seen before
                new_rule = self.rule_finder(X, Y, W, target_class, domain )

                # the general requirements can be found
                if len(rule_list) >= max_iteration:
                    print("max number rule generated. exiting now")
                    break
                # if new_rule == None:
                #     print("None Rule Generated")
                #     break
                # exclusive removed the covered instance of target class
                X, Y, W = self.cover_and_remove(X, Y, W, new_rule)
                rule_list.append(new_rule)
                if verbose:
                    pbar.update(1)
            pbar.close()

            return rule_list

        def fit(self, X, Y, domain,target_class,W=None,number = 10):
            self.domain = domain
            Y = Y.astype(dtype=int)
            rule_list = []

            target_class_idx = domain.class_var.values.index(target_class)

            # rule_list.extend(self.find_rules(X, Y, W, target_class_idx,self.domain))
            rule_list = self.find_rules(X, Y, W, target_class_idx,self.domain,number=number)
            # add the default rule
            # rule_list.append(self.generate_default_rule(X, Y, W, self.domain))
            return rule_list


    for feature in data_table.domain.attributes:
        if feature.is_continuous:
            feature.max = max([ d[feature] for d in data_table ]).value
            feature.min = min([ d[feature] for d in data_table ]).value

    # fit the explainer to the data
    explainer = cn2anchor_explainer(data_table,blackbox)
    rule_list = explainer.fit(data_table.X,data_table.Y,data_table.domain,target_class=target_class,number=number)

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
