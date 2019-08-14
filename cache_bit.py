import sys
# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

this.cache_bit_map_condition = {}
this.cache_bit_map_condition_size = {}
this.cache_bit_map_rule = {}
this.cache_bit_map_rule_size = {}

import numpy as np

class Cache_Condition_Bit():

    def __init__(self):
        return

    def initialize(self,X):
        self.X_size = X.shape[0]
        this.cache_bit_map_condition = {}
        this.cache_bit_map_condition_size = {}
        return

    def filter_data(self,condition,X):
        if X.shape[0]>self.X_size:
            self.X_size = X.shape[0]
        if self.X_size == X.shape[0]:
            if condition not in this.cache_bit_map_condition:
                this.cache_bit_map_condition[condition] = condition.filter_data_(X)
                this.cache_bit_map_condition_size[condition] = X.shape[0]
            elif X.shape[0] > this.cache_bit_map_condition_size[condition]:
                old = this.cache_bit_map_condition[condition]
                old_size = this.cache_bit_map_condition_size[condition]
                new = condition.filter_data_(X[old_size:])
                this.cache_bit_map_condition[condition] = np.concatenate( (old,new) )
                this.cache_bit_map_condition_size[condition] = X.shape[0]
            return this.cache_bit_map_condition[condition]
        else:
            return condition.filter_data_(X)
        # if self.X_size == X.shape[0]:
        #     if condition not in this.cache_bit_map_condition:
        #         this.cache_bit_map_condition[condition] = condition.filter_data_(X)
        #         this.cache_bit_map_condition_size[condition] = X.shape[0]
        #     return this.cache_bit_map_condition[condition]
        # elif X.shape[0] > self.X_size:
        #     if condition not in this.cache_bit_map_condition:
        #         this.cache_bit_map_condition[condition] = condition.filter_data_(X)
        #         this.cache_bit_map_condition_size[condition] = X.shape[0]
        #     old = this.cache_bit_map_condition[condition]
        #     old_size = this.cache_bit_map_condition_size[condition]
        #     new = condition.filter_data_(X[old_size:])
        #     return np.concatenate( (old,new) )
        # elif X.shape[0]<self.X_size:
        #     return condition.filter_data_(X)

class Cache_Rule_Bit():

    def __init__(self):
        return

    def initialize(self,X):
        self.X_size = X.shape[0]
        this.cache_bit_map_rule = {}
        this.cache_bit_map_rule_size = {}
        return

    def evaluate_data(self,rule,X):
        if X.shape[0]>self.X_size:
            self.X_size = X.shape[0]

        if self.X_size == X.shape[0]:
            # key = rule.string_representation
            key = rule
            if key not in this.cache_bit_map_rule:
                this.cache_bit_map_rule[key] = rule.evaluate_data_(X)
                this.cache_bit_map_rule_size[key] = X.shape[0]
            elif X.shape[0] > this.cache_bit_map_rule_size[key]:
                old = this.cache_bit_map_rule[key]
                old_size = this.cache_bit_map_rule_size[key]
                new = rule.evaluate_data_(X[old_size:])
                this.cache_bit_map_rule[key] = np.concatenate( (old,new) )
                this.cache_bit_map_rule_size[key] = X.shape[0]

            return this.cache_bit_map_rule[key]
        else:
            return rule.evaluate_data_(X)
        # key = rule
        # if self.X_size == X.shape[0]:
        #     # key = rule.string_representation
        #     if key not in this.cache_bit_map_rule:
        #         this.cache_bit_map_rule[key] = rule.evaluate_data_(X)
        #         this.cache_bit_map_rule_size[key] = X.shape[0]
        #     return this.cache_bit_map_rule[key]
        # elif X.shape[0] > self.X_size:
        #     if key not in this.cache_bit_map_rule:
        #         this.cache_bit_map_rule[key] = rule.evaluate_data_(X[:self.X_size])
        #         this.cache_bit_map_rule_size[key] = self.X_size
        #     old = this.cache_bit_map_rule[key]
        #     old_size = this.cache_bit_map_rule_size[key]
        #     new = rule.evaluate_data_(X[old_size:])
        #     return np.concatenate( (old,new) )
        # elif X.shape[0] < self.X_size:
        #     return rule.evaluate_data_(X)
