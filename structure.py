'''
this struture.py contains three important data structure
1. Condition: represents a sinle condition
2. Rule: represent a single rule
3. Action: an action to a given rule set
'''

import operator
import math
import numpy as np

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
