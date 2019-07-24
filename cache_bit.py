import sys
# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

this.cache_bit_map_condition = {}
this.cache_bit_map_rule = {}

class Cache_Condition_Bit():

    def __init__(self):
        return

    def initialize(self,X):
        self.X_size = X.shape[0]
        this.cache_bit_map_condition = {}
        return

    def filter_data(self,condition,X):
        if self.X_size == X.shape[0]:
            if condition not in this.cache_bit_map_condition:
                this.cache_bit_map_condition[condition] = condition.filter_data_(X)
            return this.cache_bit_map_condition[condition]
        else:
            return condition.filter_data_(X)

class Cache_Rule_Bit():

    def __init__(self):
        return

    def initialize(self,X):
        self.X_size = X.shape[0]
        this.cache_bit_map_rule = {}
        return

    def evaluate_data(self,rule,X):
        try:
            if self.X_size == X.shape[0]:
                # key = rule.string_representation
                key = rule
                if key not in this.cache_bit_map_rule:
                    this.cache_bit_map_rule[key] = rule.evaluate_data_(X)
                return this.cache_bit_map_rule[key]
            else:
                return rule.evaluate_data_(X)
        except:
            print(rule)
            quit()
