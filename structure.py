'''
this struture.py contains three important data structure
1. Condition: represents a sinle condition
2. Rule: represent a single rule
3. Action: an action to a given rule set
'''

import operator
import math
import numpy as np
import random
from copy import deepcopy

import Orange
from Orange.data import Table
from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot



from core import simple_objective, get_incorrect_cover_ruleset
from utils import itemsets_to_rules,rule_to_string,add_condition

# this is a hyperparameter how specific a value could be.
# TODO: explains more on it
CONTINUOUS_PERCENTAGE_THRESHOLD = 0.01



class Condition():
    '''
    a single condition.
    It is modified based on the condition class from Orange package. (in particualr, Orange.classification.rule)
    '''
    OPERATORS = {
        # discrete, nominal variables
        'in': lambda x,y: operator.contains(y,x),
        # continuous variables
        '<=': operator.le,
        '>=': operator.ge,
    }
    def __init__(self, column, values=None, min_value=-math.inf,max_value=math.inf,type='categorical',possible_range=[0,100]):
        '''
        differs in the number of arguments as for categorical feature and continuous features
        '''
        self.type = type
        self.column = column
        if type == 'categorical':
            # for categorical features
            self.values = values # value is a set of possible values
        elif type == 'continuous':
            self.min = max(min_value,possible_range[0]) # for continuous features
            self.max = min(max_value,possible_range[1]) # for continuous features
        else:
            print("critical error. Type of condition unspecified. Must be one of categorical or continuous ")

    def filter_instance(self, x):
        """
        Filter a single instance. Returns true or false
        """
        if self.type == 'categorical':
            return Condition.OPERATORS['in'](x[self.column], self.values)
        elif self.type == 'continuous':
            return Condition.OPERATORS['<='](x[self.column], self.max) and Condition.OPERATORS['>='](x[self.column], self.min)

    def filter_data(self, X):
        """
        Filter several instances concurrently. Retunrs array of bools
        """
        if self.type == 'categorical':
            return Condition.OPERATORS['in'](X[:,self.column], self.values)
        elif self.type == 'continuous':
            return np.logical_and(
            Condition.OPERATORS['<='](X[:,self.column], self.max),
            Condition.OPERATORS['>='](X[:,self.column], self.min)
            )

    def __eq__(self, other):
        # return self.selectors == other.selectors
        if self.type != other.type:
            raise Exception('two selector should have the same type. The value of s1 was: {0} and the type of s2 was: {1}'.format(self.type,other.type))
            return

        if self.type == 'categorical':
            return self.values == other.values
        elif self.type == 'continuous':
            return self.min == other.min and self.max == other.max
        else:
            raise Exception('critical error: unknown selector type {}'.format(self.type))
            return


    def modify_to_cover(self,x,domain):
        '''
        Given an instance that is not covered, modify this conditon to cover it
        '''
        if self.type == 'categorical':
            exact_value = x[self.column]
            if exact_value not in domain.attributes[self.column].values:
                raise ValueError("unknow error happens in modify_to_cover.. The value is not fitted in the domain: the exact value is ",exact_value," and the domain of this feature is",domain.attributes[self.column].values)
            self.values.append(exact_value)
        elif self.type == 'continuous':
            exact_value = x[self.column]
            if exact_value > self.max:
                # TODO
                # self.max = exact_value + (exact_value-self.max)*0.01
                exact_value = self.round_continuous(exact_value,domain.attributes[self.column].max,domain.attributes[self.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD,mode='floor')
                self.max = exact_value
                self.max = min(self.max,domain.attributes[self.column].max)
            elif exact_value < self.min:
                # TODO
                # self.min = exact_value + (exact_value-self.min)*0.01
                exact_value = self.round_continuous(exact_value,domain.attributes[self.column].max,domain.attributes[self.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD,mode='ceil')
                self.min = exact_value
                self.min = max(self.min,domain.attributes[self.column].min)
            else:
                print(self.column,self.min,self.max)
                print()
                raise ValueError("unknow error, happens in modify_to_cover. the exact value is ",exact_value, "current condition value is (min,max)", self.min, self.max ," and the domain of this feature is (max)",domain.attributes[self.column].max,"(min)",domain.attributes[self.column].min)
        else:
            raise ValueError("not possible, type either categorical or continuous")

    def round_continuous(self,value,max,min,percentage_bin,mode='normal'):
        '''
        A helper function to round continuous values into a minimum threshold, 1% percentage of the domain interval. This percentage is defined as a hyper-parameter
        For example, if for a domain we have max =100, and min = 0
        then the values of continuous value will have a smallest unit of 1% (100-0), which is 1.
        We will have 54, 32, etc but not 54.5. Since the smallest unit is 1.
        '''
        if mode == 'floor':
            n_percentage = math.floor( (value-min) * 1.0 / (percentage_bin*(max-min)) )
        elif mode == 'ceil':
            n_percentage = math.ceil( (value-min) * 1.0 / (percentage_bin*(max-min)) )
        elif mode == 'normal':
            n_percentage = int( (value-min) * 1.0 / (percentage_bin*(max-min)) )

        else:
            raise ValueError("not possible, mode must be floor or ceil but get",mode)
        return n_percentage * percentage_bin * (max-min) + min

    def modify_to_not_cover(self,x,domain):
        '''
        Given an instance that is covered, modify this conditon to not cover it
        '''
        if self.type == 'categorical':
            exact_value = x[self.column]
            if exact_value not in domain.attributes[self.column].values:
                raise ValueError("unknow error happens in modify_to_not_cover.. The value is not fitted in the domain: the exact value is ",exact_value," and the domain of this feature is",domain.attributes[self.column].values)
            self.values.remove(exact_value)
        elif self.type == 'continuous':
            exact_value = x[self.column]
            # round continuous variable into minimum threshold
            exact_value = self.round_continuous(exact_value,domain.attributes[self.column].max,domain.attributes[self.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD)
            if exact_value > (self.max+self.min)/2:
                # TODO
                # self.max = exact_value + (self.max-exact_value)*0.01
                self.max = exact_value
                self.max = min(self.max,domain.attributes[self.column].max)
            elif exact_value <= (self.max+self.min)/2:
                # TODO
                # self.min = exact_value + (self.min-exact_value)*0.01
                self.min = exact_value
                self.min = max(self.min,domain.attributes[self.column].min)
            else:
                raise ValueError("unknow error, happens in modify_to_not_cover. the exact value is ",exact_value," and the domain of this feature is (max)",domain.attributes[self.column].max,"(min)",domain.attributes[self.column].min)
        else:
            raise ValueError("not possible, type either categorical or continuous")

class Rule:
    def __init__(self,conditions,domain,target_class):
        """
        Parameters:
            feature_value_pairs : a list of (column_idx,value) pairs
            class_label: the target class
            domain: the domain
        """
        self.conditions = conditions
        self.target_class = target_class
        self.target_class_idx = domain.class_var.values.index(target_class)

    def evaluate_instance(self, x):
        """
        Evaluate a single instance.

        Parameters
        ----------
        x : ndarray
            Evaluate this instance.

        Returns
        -------
        return : bool
            True, if the rule covers 'x'.
        """
        return all(condition.filter_instance(x) for condition in self.conditions)

    def evaluate_data(self, X):
        """
        Evaluate several instances concurrently.

        Parameters
        ----------
        X : ndarray
            Evaluate this data.

        Returns
        -------
        return : ndarray, bool
            Array of evaluations.
        """
        curr_covered = np.ones(X.shape[0], dtype=bool)
        for condition in self.conditions:
            curr_covered &= condition.filter_data(X)
        return curr_covered

    def __len__(self):
        return len(self.conditions)

    # def all_predicates_same(self, r):
    #     return self.itemset == r.itemset
    #
    # def class_label_same(self,r):
    #     return self.class_label == r.class_label
    #
    # def set_class_label(self,label):
    #     self.class_label = label

    def get_length(self):
        return len(self.conditions)

    def get_cover(self, X):
        res = self.evaluate_data(X)
        return np.where(res == True)[0]

    def get_correct_cover(self, X, Y):
        indices_instance_covered = self.get_cover(X)
        y_tmp = Y[indices_instance_covered]

        correct_cover = []
        for ind in range(0,len(indices_instance_covered)):
            if y_tmp[ind] == self.target_class_idx:
                correct_cover.append(indices_instance_covered[ind])

        return correct_cover, indices_instance_covered

    def get_incorrect_cover(self, X, Y):
        correct_cover, full_cover = self.get_correct_cover(X, Y)
        return (sorted(list(set(full_cover) - set(correct_cover))))

    def __eq__(self, other):
        return self.conditions == other.conditions

class DecisionSet():
    def __init__(self,data_table,target_class='yes',seed=42):
        random.seed(seed)
        self.data_table = data_table
        self.domain = data_table.domain
        # for continuous variabels, we compute max and min
        for feature in self.domain.attributes:
            if feature.is_continuous:
                feature.max = max([ d[feature] for d in self.data_table ])
                feature.min = min([ d[feature] for d in self.data_table ])
        self.target_class = target_class
        # for example target_class = 'yes' and the domain Y is ['no','yes']
        # then the target class idx = 1
        self.target_class_idx = self.data_table.domain.class_var.values.index(self.target_class)
        return


    def generate_action(self,X,Y):
        incorrect_instances = get_incorrect_cover_ruleset(self.current_solution,X,Y)
        anchor_instance = random.choice(incorrect_instances)
        N = len(self.current_solution)
        t = random.random()
        if Y[anchor_instance]==1 or N<=1:
            if t<1.0/3 or N <= 1:
                mode = 'ADD_RULE'       # action: add rule
            elif t < 2.0/3 :
                mode = 'REMOVE_CONDITION' # action: replace
            else:
                mode = 'EXPAND_CONDITION'
        else:
            if t<1.0/2:
                mode = 'REMOVE_RULE'       # action: remove rule
            elif t < 2.0/3 :
                mode = 'ADD_CONDITION' # action: replace
            else:
                mode = 'SPECIFY_CONDITION'

        actions = []
        self.current_obj = simple_objective(self.current_solution,X,Y)

        if mode == 'ADD_RULE':
            if not hasattr(self, 'rule_space'):
                self.generate_rule_space()
            for candidate_rule_to_add in self.rule_space:
                # if candidate_rule.filter_instance(X[anchor_instance]) == True:
                #     action = Action(self.current_solution,('ADD_RULE',candidate_rule),X,Y,domain=self.domain,current_obj = self.current_obj)
                #     actions.append(action)
                # else:
                #     continue
                if candidate_rule_to_add in self.current_solution:
                    continue
                action = Action(self.current_solution,('ADD_RULE',candidate_rule_to_add),X,Y,domain=self.domain,current_obj = self.current_obj)
                actions.append(action)
        elif mode == 'REMOVE_RULE':
            for candidate_rule_to_remove in self.current_solution:
                action = Action(self.current_solution,('REMOVE_RULE',candidate_rule_to_remove),X,Y,domain=self.domain,current_obj = self.current_obj)
                actions.append(action)
        elif mode == 'REMOVE_CONDITION':
            for rule_idx,rule in enumerate(self.current_solution):
                for condition_idx,condition in enumerate(rule.conditions):
                    rule_new = deepcopy(rule)
                    del rule_new.conditions[condition_idx]
                    if len(rule_new.conditions) == 0:
                        # which means there is only one condition in this rule, remove this condition means to remove the entire rule
                        action = Action(self.current_solution,('REMOVE_RULE',rule),X,Y,domain=self.domain,current_obj = self.current_obj)
                    else:
                        action = Action(self.current_solution,('REPLACE_RULE',rule,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj)
                    actions.append(action)
        elif mode == 'ADD_CONDITION':
            for rule_idx,rule in enumerate(self.current_solution):
                used_attribute_columns = [ c.column for c in rule.conditions]
                for attribute_idx,attribute in enumerate(self.domain.attributes):
                    if attribute_idx in used_attribute_columns:
                        continue
                    if attribute.is_discrete:
                        for possible_value in attribute.values:
                            print("------",possible_value)
                            rule_new = deepcopy(rule)
                            add_condition(rule_new,rule_idx,attribute_idx,value)
                            action = Action(self.current_solution,('REPLACE_RULE',rule,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj)
                            actions.append(action)
                    elif attribute.is_continuous:
                        # we use the discretized bin for continuous.
                        # the difference is trivial compared wtih estimating use info gain here.
                        # since we could always modify a condition using other actions.
                        for possible_value in self.disc_data_table.domain.attributes[attribute_idx].values:
                            rule_new = deepcopy(rule)
                            add_condition(rule_new,rule_idx,attribute_idx,possible_value,self.domain)
                            action = Action(self.current_solution,('REPLACE_RULE',rule,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj)
                            actions.append(action)
        elif mode == 'EXPAND_CONDITION':
            # try:
            for rule in self.current_solution:
                for condition_idx,condition in enumerate(rule.conditions):
                    if condition.filter_instance(X[anchor_instance]) == True:
                        # then we do not have to modify this condition
                        continue
                    rule_new = deepcopy(rule)
                    rule_new.conditions[condition_idx].modify_to_cover(X[anchor_instance],self.domain)
                    action = Action(self.current_solution,('REPLACE_RULE',rule,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj)
                    actions.append(action)
            # except:
            #     for e in self.current_solution:
            #         print(rule_to_string(e,self.domain,self.target_class_idx))
            #     print("dddd")
            #     print(rule_to_string(rule_new,self.domain,self.target_class_idx) )
            #     print(X[anchor_instance])
            #     quit()

        elif mode == 'SPECIFY_CONDITION':
            for rule in self.current_solution:
                for condition_idx,condition in enumerate(rule.conditions):
                    rule_new = deepcopy(rule)
                    rule_new.conditions[condition_idx].modify_to_not_cover(X[anchor_instance],self.domain)
                    action = Action(self.current_solution,('REPLACE_RULE',rule,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj)
                    actions.append(action)
        else:
            raise ValueError("not implement this mode:", mode)

        return actions




    def generate_rule_space(self):
        """
        using fp-growth to generate the space of rules.
        note that it needs itemset, so the first step is to translate the data in the form of item set
        """
        self.maxlen = 10
        self.supp = 2

        positive_indices = np.where(self.data_table.Y == self.target_class_idx)[0]

        # first discretize the data
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=5)
        # disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
        # disc.method = Orange.preprocess.discretize.EqualWidth(n=3)
        disc_data_table = disc(self.data_table[positive_indices])
        self.disc_data_table = disc_data_table

        X,mapping = OneHot.encode(disc_data_table,include_class=False)
        itemsets = list(frequent_itemsets(X,self.supp))
        decoder_item_to_feature = { idx:(feature.name,value) for idx,feature,value in OneHot.decode(mapping,disc_data_table,mapping)}

        self.rule_space = itemsets_to_rules(itemsets,self.domain,decoder_item_to_feature,self.target_class)

        # TODO: print pre-mined rules
        # for r in self.rule_space:
        #     print(rule_to_string(r,self.domain,1))


class Action():

    def __init__(self,current_solution,modification,X,Y,domain,current_obj):
        mode = modification[0]
        self.current_solution = current_solution
        self.current_obj = current_obj
        self.hopeless = False
        if mode == 'ADD_RULE':
            _,add_rule = modification
            self.new_solution = deepcopy(self.current_solution); self.new_solution.append(add_rule)
            self.new_obj = simple_objective(self.new_solution,X,Y)
        elif mode == 'REMOVE_RULE':
            _,remove_rule = modification
            self.new_solution = deepcopy(self.current_solution); self.new_solution.remove(remove_rule)
            self.new_obj = simple_objective(self.new_solution,X,Y)
        elif mode == 'REPLACE_RULE':
            _,old_rule,new_rule = modification
            self.new_solution = deepcopy(self.current_solution)
            self.new_solution.remove(old_rule)
            self.new_solution.append(new_rule)
            self.new_obj = simple_objective(self.new_solution,X,Y)
        else:
            raise ValueError("do not know this action mode:<",mode,">. Please check, or this mode is not implemented")

    def make_hopeless(self):
        self.hopeless = True
        self.new_obj = -10000

    def obj_estimation(self):
        return self.new_obj

    def __lt__(self,other):
        '''
        this less than comparision is crucial... make sure to implement it.
        As it will be used to select the best and second best action. as in the function in `core.py`
        '''
        return self.obj_estimation() < other.obj_estimation()

    def interval(self):
        return 0

    def upper(self):
        return self.obj_estimation() + self.interval()
    def lower(self):
        return self.obj_estimation() + self.interval()
