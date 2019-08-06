'''
this struture.py contains three important data structure
1. Condition: represents a sinle condition
2. Rule: represent a single rule
3. Action: an action to a given rule set
'''

import operator
from functools import reduce
import math
import numpy as np
import random
from copy import deepcopy,copy
import re
from sklearn.metrics.pairwise import euclidean_distances as distance_function
import heapq

import Orange
from Orange.data import Table

# todo:remove associate rule
# from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot

from core import simple_objective, get_incorrect_cover_ruleset
from utils import rule_to_string

import logging
import time

# this is a hyperparameter how specific a value could be.
# TODO: explains more on it
CONTINUOUS_PERCENTAGE_THRESHOLD = 0.05

# this is a pointer to the module object instance itself.

from cache_bit import Cache_Condition_Bit,Cache_Rule_Bit
import sys
this = sys.modules[__name__]
this.cache_condition_bit = Cache_Condition_Bit()
this.cache_rule_bit = Cache_Rule_Bit()
# Cache_rule_bit = {}
# Cache_rule_bit_Size = 0
OPERATORS = {
    # discrete, nominal variables
    # 'in': lambda x,y: all( [operator.contains(y,x_tmp) for x_tmp in x ],
    'in': lambda x,y: operator.contains(y,x),
    # continuous variables
    '<=': operator.le,
    '>=': operator.ge,
}

this.DOMAIN = None


class Condition():
    '''
    a single condition.
    It is modified based on the condition class from Orange package. (in particualr, Orange.classification.rule)
    '''
    def __init__(self, column, values=None, min_value=-math.inf,max_value=math.inf,type='categorical',domain=None):
        '''
        differs in the number of arguments as for categorical feature and continuous features
        '''
        self.type = type
        self.column = column
        if type == 'categorical':
            # for categorical features
            self.values = values[:] # value is a set of possible values
        elif type == 'continuous':
            if domain == None:
                max_limit = this.DOMAIN.attributes[column].max
                min_limit = this.DOMAIN.attributes[column].min
            else:
                max_limit = domain.attributes[column].max
                min_limit = domain.attributes[column].min
            min_value,max_value = sorted([min_value,max_value])
            self.min = max(min_value,min_limit) # for continuous features
            self.max = min(max_value,max_limit) # for continuous features
            if self.min > self.max:
                print("error!!!min and max")
                print(self.min,self.max)
                quit()
        else:
            print("critical error. Type of condition unspecified. Must be one of categorical or continuous ")


    def filter_instance(self, x):
        """
        Filter a single instance. Returns true or false
        """
        if self.type == 'categorical':
            return OPERATORS['in'](x[self.column], self.values)
        elif self.type == 'continuous':
            return OPERATORS['<='](x[self.column], self.max) and OPERATORS['>='](x[self.column], self.min)

    def filter_data(self, X):
        """
        Filter several instances concurrently. Retunrs array of bools
        Note that this is the cached version
        """
        if this.cache_condition_bit is None:
            return self.filter_data_(X)
        result = this.cache_condition_bit.filter_data(self,X)
        # if not np.array_equal(result,self.filter_data_(X)):
        #     print("strange!!!")
        #     quit()
        if result is not None:
            return result
        else:
            print("strange! Should not enter here!")
            quit()

    def filter_data_(self,X):
    # def filter_data(self,X):
        """
        Filter several instances concurrently. Retunrs array of bools
        this is the not cached true filter_data method
        """
        if self.type == 'categorical':
            # return OPERATORS['in'](X[:,self.column],self.values)
            try:
                tmp = np.stack([np.equal(X[:,self.column], v) for v in self.values ])
                return np.any(tmp,axis=0)
                # if tmp.shape[0]>2:
                #     print("ppp")
                #     print(tmp.shape)
                #     print(np.any(tmp,axis=0).shape)
                #     print(np.any(tmp,axis=0)[:5])
                #     print( np.logical_or.reduce(tmp).shape)
                #     print( np.logical_or.reduce(tmp)[:5])
                #     print("ppp")
                #     quit()
                # return np.logical_or.reduce(tmp)
                # return [ (x[self.column] in self.values) for x in X ]
            except:
                print(self.values)
                print(X[:5,self.column])
                print(np.logical_or.reduce(
                [np.equal(X[:5,self.column], v) for v in self.values ]
                ))
                quit()
        elif self.type == 'continuous':
            return np.logical_and(
            OPERATORS['<='](X[:,self.column], self.max),
            OPERATORS['>='](X[:,self.column], self.min)
            )
            # return [ (x[self.column] <= self.max ) and (x[self.column >= self.min ) for x in X ]

    def __eq__(self, other):
        # return self.selectors == other.selectors

        if self.type != other.type:
            # raise Exception('two selector should have the same type. The value of s1 was: {0} and the type of s2 was: {1}'.format(self.type,other.type))
            return False

        if self.type == 'categorical':
            self.values = sorted(self.values);
            other.values = sorted(other.values);
            return self.values == other.values
        elif self.type == 'continuous':
            return self.min == other.min and self.max == other.max
        else:
            raise Exception('critical error: unknown selector type {}'.format(self.type))
            return

    def __hash__(self):
        if self.type == 'categorical':
            self.values = sorted(self.values);
            return hash((self.type, self.column ,tuple(self.values)) )
        elif self.type == 'continuous':
            return hash((self.type,self.column, self.min,self.max) )
        else:
            raise Exception('critical error: unknown selector type {}'.format(self.type))
            return
        return

    def __deepcopy__(self,memodict={}):
        # copy_object = Condition()
        # copy_object.value = self.value
        if self.type == 'categorical':
            column = self.column
            values = self.values[:]
            return Condition(column=column,values=values,type='categorical')
            # return Condition(column=deepcopy(self.column,memodict),values=deepcopy(self.values,memodict),type='categorical')
        elif self.type == 'continuous':
            return  Condition(column=self.column,min_value=self.min,max_value=self.max,type='continuous')
        else:
            print("critical error. deep copy error! __deepcopy__ for Condition ")
            quit()

    def __sub__(self,other):
        if self.column != other.column:
            return deepcopy(self)
        else:
            # assert self.type == other.type,"Assertion Error: different type (categ,contin)"
            if self.type == 'categorical':
                new = deepcopy(self)
                # if is included or equal then just return it
                if self.values == other.values:
                    return new
                elif set(self.values).issubset( set(other.values) ):
                    # this indicate the strict subset
                    return new
                else:
                    for value in other.values:
                        if value in self.values:
                            new.values.remove(value)
                return new
            else:
                if self.max <= other.min or self.min >= other.max:
                    return deepcopy(self)
                elif self.max <= other.max and self.min >= other.min:
                    return deepcopy(self)
                elif self.max <= other.max:
                    new = deepcopy(self)
                    new.max = other.min
                    return new
                elif self.min >= other.min:
                    new = deepcopy(self)
                    new.min = other.max
                    return new
                else:
                    # print("this is tricky problem. Operator being subtracted is",self.min,self.max,"yet the substractor is",other.min,other.max)
                    # print("using a lazy update: see structure. Condition.__sub__ function")
                    new = deepcopy(self)
                    return new

        return

    def sample(self,batch_size=10):
        if self.type == 'categorical':
            synthetic_column = np.random.choice(self.values,size=batch_size)
            return synthetic_column
        elif self.type == 'continuous':
            if not isinstance(batch_size, int):
                print("strange thing happend! it is not a int:",batch_size)
            synthetic_column = np.random.uniform(low=self.min,high=self.max,size=batch_size)
            # synthetic_column = np.random.uniform(low=self.min,high=self.max,size=10)
            return synthetic_column

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

    def compute_volume(self,domain):
        if self.type == 'categorical':
            return len(self.values) * 1.0 / len(domain.attributes[self.column].values)
        elif self.type == 'continuous':
            return (self.max-self.min) / (domain.attributes[self.column].max-domain.attributes[self.column].min)


class Rule:
    def __init__(self,conditions,domain,target_class_idx,string_representation=None):
        """
        Parameters:
            feature_value_pairs : a list of (column_idx,value) pairs
            class_label: the target class
            domain: the domain
        """
        # self.conditions = conditions
        self.conditions = sorted(conditions,key=lambda x:x.column)
        for c in self.conditions:
            if c.type == 'categorical':
                c.values = sorted(c.values)
        self.target_class_idx = target_class_idx
        # self.compute_volume(domain)
        if string_representation == None:
            self.string_representation = rule_to_string(self,domain=domain,target_class_idx=self.target_class_idx)
        else:
            self.string_representation = string_representation

    def compute_volume(self,domain):
        self.volume = reduce(operator.mul, [ c.compute_volume(domain) for c in self.conditions] , 1)
        return self.volume

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
        if this.cache_rule_bit is None:
            return self.evaluate_data_(X)
        result = this.cache_rule_bit.evaluate_data(self,X)
        # if not np.array_equal(result,self.evaluate_data_(X)):
        #     print("strange!!!")
        #     quit()
        if result is not None:
            return result
        else:
            print("strange! Should not enter here!")
            quit()

    def evaluate_data_(self, X):
    # def evaluate_data(self, X):
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
        # curr_covered = np.ones(X.shape[0], dtype=bool)
        # for condition in self.conditions:
        #     curr_covered &= condition.filter_data(X)
        try:
            tmp = np.stack([condition.filter_data(X) for condition in self.conditions])
            # curr_covered = np.logical_and.reduce(tmp)
            curr_covered = np.all(tmp,axis=0)
            # quit()
            return curr_covered
        except:
            print(len(self.conditions))
            print(self.string_representation)
            print([len(c.values) for c in self.conditions])
            print("*".join([str(c.values) for c in self.conditions]))
            print([t.shape for t in tmp])
            quit()

    def __len__(self):
        return len(self.conditions)

    def __hash__(self):
        # self.conditions = sorted(self.conditions,key=lambda x:x.column)
        #
        # l = [len(self.conditions) ]+ [ hash(c) for c in self.conditions ]
        # return hash(tuple(l))
        return hash(self.string_representation)

    def __eq__(self, other):
        # if len(self.conditions) != len(other.conditions):
        #     return False
        # else:
        #     self.conditions = sorted(self.conditions,key=lambda x:x.column)
        #     other.conditions = sorted(other.conditions,key=lambda x:x.column)
        #     return all([ a==b for a,b in zip(self.conditions,other.conditions) ] )
        return self.string_representation == other.string_representation

    def __sub__( self, other ):
        # conditions = deepcopy(self.conditions)
        conditions = []
        for i,(c,tmp_c) in enumerate(zip(self.conditions,other.conditions)):
            conditions.append(c-tmp_c)
        # for i,c in enumerate(self.conditions):
        #     for tmp_c in other.conditions:
        #         conditions[i] = c - tmp_c
        new_rule = deepcopy(self)
        new_rule.conditions = conditions
        return new_rule

    def __lt__(self,other):
        # return hash(self) < hash(other)
        return self.string_representation <= other.string_representation

    def __deepcopy__(self,memodict={}):
        return Rule(conditions=[ deepcopy(c,memodict) for c in self.conditions],domain=this.DOMAIN,target_class_idx=self.target_class_idx,string_representation=self.string_representation)

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

    def add_condition(self,rule_idx,attribute_idx,value,domain):
        '''
        add a condition into a rule
        '''
        attribute = domain.attributes[attribute_idx]
        col_idx = attribute_idx
        if attribute.is_discrete:
            s = Condition(column=col_idx,values=[value],type='categorical')

        elif attribute.is_continuous:
            c = re.split( "- | ",value)
            if len(c) == 2:
                if c[0] in ['<=','≤','<']:
                    s = Condition(column=col_idx,max_value=float(c[1]),type='continuous')
                else:
                    s = Condition(column=col_idx,min_value=float(c[1]),type='continuous')

            if len(c) == 3:
                assert float(c[0]) < float(c[2]), "Add condition Assertation error!"
                s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous')
        else:
            raise ValueError('not possible, must be discrete or continuous:',attribute)
        # new_conditions = [ deepcopy(c) for c in self.conditions] +[s]
        new_conditions = self.conditions +[s]
        return Rule(conditions=new_conditions,domain=domain,target_class_idx=self.target_class_idx)

    def remove_condition(self,condition_idx,domain):
        '''remove condition and return a new rule'''
        # conditions= [deepcopy(c) for i,c in enumerate(self.conditions) if i!=condition_idx]
        conditions= [ c for i,c in enumerate(self.conditions) if i!=condition_idx]
        return Rule(conditions=conditions,domain=domain,target_class_idx=self.target_class_idx)

    def modify_to_cover(self,condition_idx,x,domain):
        '''
        Given an instance that is not covered, modify this conditon to cover it,
        return (the_new_rule,modify_possible)
        the second element is True if modify successful, else False
        '''
        condition = self.conditions[condition_idx]
        if condition.type == 'categorical':
            exact_value = x[condition.column]
            if exact_value not in range(len(domain.attributes[condition.column].values)):
                raise ValueError("unknow error happens in modify_to_not_cover.. The value is not fitted in the domain: the exact value is of index ",exact_value," and the domain of this feature is",domain.attributes[condition.column].values,"in total",len(domain.attributes[condition.column].values),"values ranging from 0")
            if exact_value not in condition.values:
                # new_condition = condition.copy()
                new_condition =  Condition(column=condition.column,values=condition.values+[exact_value],type='categorical')
            else:
                return None,False
        elif condition.type == 'continuous':
            # new_condition = condition.copy()
            exact_value = x[condition.column]
            if exact_value > condition.max:
                # TODO
                # self.max = exact_value + (exact_value-self.max)*0.01
                exact_value = condition.round_continuous(exact_value,domain.attributes[condition.column].max,domain.attributes[condition.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD,mode='floor')
                max_value = min(exact_value,domain.attributes[condition.column].max)
                new_condition = Condition(column=condition.column,max_value=max_value,min_value=condition.min,type='continuous')
            elif exact_value < condition.min:
                # TODO
                # self.min = exact_value + (exact_value-self.min)*0.01
                exact_value = condition.round_continuous(exact_value,domain.attributes[condition.column].max,domain.attributes[condition.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD,mode='ceil')
                min_value = max(exact_value,domain.attributes[condition.column].min)
                new_condition = Condition(column=condition.column,max_value=condition.max,min_value=min_value,type='continuous')
            else:
                print(condition.column,condition.min,condition.max)
                print()
                raise ValueError("unknow error, happens in modify_to_cover. the exact value is ",exact_value, "current condition value is (min,max)", condition.min, condition.max ," and the domain of this feature is (max)",domain.attributes[condition.column].max,"(min)",domain.attributes[condition.column].min)
        else:
            raise ValueError("not possible, type either categorical or continuous")
        # new_conditions = [ new_condition if idx==condition_idx else deepcopy(c) for idx,c in enumerate(self.conditions)  ]
        new_conditions = [ new_condition if idx==condition_idx else c for idx,c in enumerate(self.conditions)  ]
        new_rule = Rule(conditions=new_conditions,domain=domain,target_class_idx=self.target_class_idx)
        return new_rule,True

    def modify_to_not_cover(self,condition_idx,x,domain):
        '''
        Given an instance that is covered, modify this conditon to not cover it
        return (the_new_rule,modify_possible)
        the second element is True if modify successful, else False
        '''
        condition = self.conditions[condition_idx]
        if condition.type == 'categorical':
            exact_value = x[condition.column]
            if exact_value not in range(len(domain.attributes[condition.column].values)):
                raise ValueError("unknow error happens in modify_to_not_cover.. The value is not fitted in the domain: the exact value is of index ",exact_value," and the domain of this feature is",domain.attributes[condition.column].values,"in total",len(domain.attributes[condition.column].values),"values ranging from 0")
            if exact_value in condition.values:
                new_values = condition.values[:]
                new_values.remove(exact_value)
                if len(new_values) == 0:
                    # it means it now has no possible value.
                    return None,False
                new_condition = Condition(column=condition.column,values=new_values,type='categorical')
            else:
                return None,False
        elif condition.type == 'continuous':
            exact_value = x[condition.column]
            # round continuous variable into minimum threshold
            min_value = condition.min
            max_value = condition.max
            exact_value = condition.round_continuous(exact_value,domain.attributes[condition.column].max,domain.attributes[condition.column].min,CONTINUOUS_PERCENTAGE_THRESHOLD)
            if exact_value > (condition.max+condition.min)/2:
                max_value = min(exact_value,domain.attributes[condition.column].max)
            elif exact_value <= (condition.max+condition.min)/2:
                min_value = max(exact_value,domain.attributes[condition.column].min)
            else:
                raise ValueError("unknow error, happens in modify_to_not_cover. the exact value is ",exact_value," and the domain of this feature is (max)",domain.attributes[condition.column].max,"(min)",domain.attributes[condition.column].min)
            new_condition = Condition(column=condition.column,min_value=min_value,max_value=max_value,type='continuous')
        else:
            raise ValueError("not possible, type either categorical or continuous")
        new_conditions = [ new_condition if idx==condition_idx else deepcopy(c) for idx,c in enumerate(self.conditions)  ]
        new_rule = Rule(conditions=new_conditions,domain=domain,target_class_idx=self.target_class_idx)
        return new_rule,True

def deepcopy_decision_set(decision_set):
    return deepcopy(decision_set)

class Decision_Set_Learner():
    def __init__(self,data_table,target_class='yes',seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.data_table = data_table
        self.domain = data_table.domain
        this.DOMAIN = data_table.domain
        # for continuous variabels, we compute max and min
        for feature in self.domain.attributes:
            if feature.is_continuous:
                feature.max = max([ d[feature] for d in self.data_table ])
                feature.min = min([ d[feature] for d in self.data_table ])
        self.target_class = target_class
        # for example target_class = 'yes' and the domain Y is ['no','yes']
        # then the target class idx = 1
        self.target_class_idx = self.data_table.domain.class_var.values.index(self.target_class)

        this.cache_condition_bit = Cache_Condition_Bit()
        this.cache_rule_bit = Cache_Rule_Bit()
        this.cache_condition_bit.initialize(data_table.X)
        this.cache_rule_bit.initialize(data_table.X)
        # this.cache_condition_bit = None
        # this.cache_rule_bit = None
        return
    def finish(self):
        this.cache_condition_bit = None
        this.cache_rule_bit = None

    def generate_action(self,X,Y,beta=1000,rho=None):
        incorrect_instances = get_incorrect_cover_ruleset(self.current_solution,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)
        if len(incorrect_instances) != 0:
            anchor_instance = random.choice(incorrect_instances)
            N = len(self.current_solution)
            t = random.random()
            anchor_instance_y = self.total_Y[anchor_instance]
            anchor_instance_x = self.total_X[anchor_instance]
            # if self.data_table.Y[anchor_instance]== self.target_class_idx or N<1:
            if anchor_instance_y == self.target_class_idx or N<1:
                # mode = 'ADD_RULE'       # action: add rule
                if t< 1.0/3 or N <1:
                    mode = 'ADD_RULE'       # action: add rule
                elif t < 2.0/3 :
                    mode = 'REMOVE_CONDITION' # action: replace
                else:
                    mode = 'EXPAND_CONDITION'
            else:
                # mode = 'REMOVE_RULE'       # action: remove rule
                if t < 1.0/3:
                    mode = 'REMOVE_RULE'       # action: remove rule
                    if N <=1:
                        mode = random.choice(['ADD_CONDITION','SPECIFY_CONDITION'])
                elif t < 2.0/3 :
                    mode = 'ADD_CONDITION' # action: replace
                else:
                    mode = 'SPECIFY_CONDITION'
        else:
            mode = random.choice(['REMOVE_RULE','REMOVE_CONDITION' ,'ADD_CONDITION'])

        self.mode = mode
        actions = []
        self.current_obj = simple_objective(self.current_solution,X,Y,target_class_idx=self.target_class_idx)

        logging.debug("the current mode of action generating is"+mode)

        if mode == 'ADD_RULE':
            # todo: remove this
            # if not hasattr(self,"rule_space"):
            #     print("using pre-mined rules")
            #     self.generate_rule_space()
            # new_rules_candidates = self.rule_space
            new_rules_candidates = self.new_rule_generating()
            if new_rules_candidates is not None:
                for candidate_rule_to_add in new_rules_candidates:
                    already_have_this_rule =False
                    for existed_rule in self.current_solution:
                        if candidate_rule_to_add == existed_rule:
                            already_have_this_rule = True
                            break
                    if already_have_this_rule == True:
                        continue
                    action = Action(self.current_solution,('ADD_RULE',candidate_rule_to_add),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                    actions.append(action)

        elif mode == 'REMOVE_RULE':
            for candidate_rule_to_remove_idx,candidate_rule_to_remove in enumerate(self.current_solution):
                action = Action(self.current_solution,('REMOVE_RULE',candidate_rule_to_remove_idx),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                actions.append(action)
            # actions = list(set(actions))
        elif mode == 'REMOVE_CONDITION':
            for rule_idx,rule in enumerate(self.current_solution):
                for condition_idx,condition in enumerate(rule.conditions):
                    rule_new = rule.remove_condition(condition_idx,self.domain)
                    if len(rule_new.conditions) == 0:
                        # which means there is only one condition in this rule, remove this condition means to remove the entire rule
                        # action = Action(self.current_solution,('REMOVE_RULE',rule_idx),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                        continue
                    else:
                        action = Action(self.current_solution,('REMOVE_CONDITION',rule,rule_idx,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                    actions.append(action)

            # actions = list(set(actions))
        elif mode == 'ADD_CONDITION':
            start_time = time.time()
            for rule_idx,rule in enumerate(self.current_solution):
                if rule.evaluate_instance(anchor_instance_x) == False:
                    continue
                used_attribute_columns = [ c.column for c in rule.conditions]
                for attribute_idx,attribute in enumerate(self.domain.attributes):
                    if attribute_idx in used_attribute_columns:
                        continue
                    if attribute.is_discrete:
                        for possible_value_idx in range(len(attribute.values)):

                            rule_new = rule.add_condition(rule_idx,attribute_idx,possible_value_idx,self.domain)
                            if rule_new.evaluate_instance(anchor_instance_x) == True:
                                continue
                            # todo: remove compute volume
                            rule_new.compute_volume(self.domain)
                            action = Action(self.current_solution,('ADD_CONDITION',rule,rule_idx,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                            actions.append(action)
                    elif attribute.is_continuous:
                        # we use the discretized bin for continuous.
                        # the difference is trivial compared wtih estimating use info gain here.
                        # since we could always modify a condition using other actions.
                        if not hasattr(self,"disc_data_table"):
                            tmp_table = Table.from_numpy(X=self.total_X,Y=self.total_Y,domain=self.domain)
                            disc = Orange.preprocess.Discretize()
                            disc.method = Orange.preprocess.discretize.EqualFreq(n=5)
                            # disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
                            # disc.method = Orange.preprocess.discretize.EqualWidth(n=3)
                            disc_data_table = disc(tmp_table)
                            self.disc_data_table = disc_data_table
                        for possible_value in self.disc_data_table.domain.attributes[attribute_idx].values:
                            rule_new = rule.add_condition(rule_idx,attribute_idx,possible_value,self.domain)
                            if rule_new.evaluate_instance(anchor_instance_x) == True:
                                continue
                            action = Action(self.current_solution,('ADD_CONDITION',rule,rule_idx,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                            actions.append(action)
            # print ('\tTook %0.3fs to generate for condition add %s actions' %( (time.time() - start_time ), str(len(actions)) )  )
            # actions = list(set(actions))
        elif mode == 'EXPAND_CONDITION':
            # try:
            for rule_idx,rule in enumerate(self.current_solution):
                for condition_idx,condition in enumerate(rule.conditions):
                    if condition.filter_instance(anchor_instance_x) == True:
                        # then we do not have to modify this condition
                        continue
                    rule_new,modify_possible = rule.modify_to_cover(condition_idx,anchor_instance_x,self.domain)
                    if modify_possible == True:
                        action = Action(self.current_solution,('EXPAND_CONDITION',rule,rule_idx,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                        actions.append(action)
                    else:
                        continue
            # actions = list(set(actions))
        elif mode == 'SPECIFY_CONDITION':
            for rule_idx,rule in enumerate(self.current_solution):
                for condition_idx,condition in enumerate(rule.conditions):
                    rule_new,modify_possible = rule.modify_to_not_cover(condition_idx,anchor_instance_x,self.domain)
                    if modify_possible == True:
                        action = Action(self.current_solution,('SPECIFY_CONDITION',rule,rule_idx,rule_new),X,Y,domain=self.domain,current_obj = self.current_obj,beta=beta,target_class_idx=self.target_class_idx,lambda_parameter=self.lambda_parameter,rho=rho)
                        actions.append(action)
                    else:
                        continue
            # actions = list(set(actions))
        else:
            raise ValueError("not implement this mode:", mode)

        # TODO: handling action error where there is no action generated
        if len(actions) <= 0:
            raise ValueError("no plausible actions, len length of actions",len(actions),"in the mode of ",mode)

        # TODO: remove this could make faster.
        # new_actions = list(set(actions))
        # if len(actions)!= len(new_actions):
        #     print("!!!! not consistent!! may exist duplicate actions")
        #     print(len(actions))
        #     print(len(new_actions))
        #
        # return new_actions
        # return list(set(actions))
        # hash_current = hash(tuple([hash(t) for t in self.current_solution]))


        actions = sorted(set(actions),key=lambda x:x.string_representation)

        assert len(set([a.mode for a in actions])) == 1, "error!!"
        return actions

    def new_rule_generating(self,beam_size=10):
        '''
        Perform the classic General-to-specfic search (TopDown) as used to find a single rule in the CN2 algorithm
        it is a warpper of function fomr the CN2 implementation from Orange
        '''
        from Orange.data import _contingency
        from Orange.classification.rules import BeamSearchAlgorithm,SearchStrategy,TopDownSearchStrategy,EntropyEvaluator,LengthEvaluator,GuardianValidator,LRSValidator,get_dist
        from Orange.classification.rules import Rule as Rule_Orange
        from Orange.preprocess.discretize import EntropyMDL,EqualFreq


        class OurTopDownSearchStrategy(TopDownSearchStrategy):

            @staticmethod
            def discretize(X, Y, W, domain):
                values, counts, other = _contingency.contingency_floatarray(
                    X, Y.astype(dtype=np.intp), len(domain.class_var.values), W)
                cut_ind = np.array(EntropyMDL(force=True)._entropy_discretize_sorted(counts.T, True))
                # print(values)
                # print(counts)
                # print(other)
                # print(cut_ind)
                # # cut_ind = np.array(EqualFreq(n=5)._entropy_discretize_sorted(counts.T, True))
                # print([values[smh] for smh in cut_ind])
                # quit()
                return [values[smh] for smh in cut_ind]

        self.search_algorithm = BeamSearchAlgorithm()
        self.search_strategy = OurTopDownSearchStrategy()

        # search heuristics
        self.quality_evaluator = EntropyEvaluator()
        self.complexity_evaluator = LengthEvaluator()
        # heuristics to avoid the over-fitting of noisy data
        self.general_validator = GuardianValidator()
        self.significance_validator = LRSValidator()

        def rcmp(rule):
            return rule.quality, rule.complexity

        curr_covered_or_not = np.zeros(self.total_X.shape[0], dtype=np.bool)
        for r in self.current_solution:
            curr_covered_or_not |= r.evaluate_data(self.total_X)
        indices = np.where( curr_covered_or_not == 0)[0]
        # these are the not covered instances
        X = self.total_X[indices]
        Y = self.total_Y[indices].astype("int")
        W = np.ones(X.shape[0])
        base_rules = []
        domain = self.data_table.domain
        target_class = self.target_class_idx
        prior_class_dist = get_dist(Y, W, domain)
        initial_class_dist = get_dist(self.total_Y.astype("int"),np.ones(self.total_X.shape[0]),domain)
        rules = self.search_strategy.initialise_rule(
            X, Y, W, target_class, base_rules, domain,
            initial_class_dist, prior_class_dist,
            self.quality_evaluator, self.complexity_evaluator,
            self.significance_validator, self.general_validator)

        if not rules:
            # return None
            return None


        rules = sorted(rules, key=rcmp, reverse=True)
        best_rules = [rules[0]]
        Best_rules = rules
        BUDGET = 100
        count_beam_search = 0
        while len(rules) > 0:
            candidates, rules = self.search_algorithm.select_candidates(rules)
            for candidate_rule in candidates:
                new_rules = self.search_strategy.refine_rule(
                    X, Y, W, candidate_rule)
                rules.extend(new_rules)
                #remove default rule from list of rules
                # if len(best_rules) == 0:
                #     best_rule = new_rules[0]
                # for new_rule in new_rules[1:]:
                #     for i,best_rule in best_rules:
                #         if (new_rule.quality > best_rule.quality and
                #                 new_rule.is_significant() ):
                #             best_rules[i] = new_rule
            rules = sorted(rules, key=rcmp, reverse=True)
            Best_rules.extend(rules)
            # Best_rules = sorted(Best_rules,key=rcmp,reverse=True)[:BUDGET]
            rules = self.search_algorithm.filter_rules(rules)

            count_beam_search+=1
            if count_beam_search >=100:
                logging.debug("more than 100 steps (conditions) for a single rule finding, break")
                break
        Best_rules = sorted(Best_rules,key=rcmp,reverse=True)[:BUDGET]
        # Best_rules = heapq.nlargest(BUDGET, Best_rules, key=rcmp)


        def Orange_rules_to_our(orange_rules,domain,target_class):
            rule_set = []
            col_names = [it.name for it in domain.attributes]
            for rule in orange_rules:
                previous_cols =[]
                conditions=[]
                if len(rule.selectors) ==0:
                    continue
                # for condition in rule_str_tuple:
                for orange_rule in rule.selectors:
                    col_idx,op,value = orange_rule[0],orange_rule[1],orange_rule[2]
                    if col_idx not in previous_cols:
                        if domain.attributes[col_idx].is_continuous:
                            if op in ['<=','≤','<']:
                                s = Condition(column=col_idx,max_value=float(value),type='continuous')
                            else:
                                s = Condition(column=col_idx,min_value=float(value),type='continuous')
                            conditions.append(s)
                        else:
                            # categorical
                            s = Condition(column=col_idx,values=[value],type='categorical')
                            conditions.append(s)
                        previous_cols.append(col_idx)
                    else:
                        s = conditions[ previous_cols.index(col_idx) ]
                        if domain.attributes[col_idx].is_continuous:
                            if op in ['<=','≤','<']:
                                s_tmp = Condition(column=col_idx,max_value=float(value),type='continuous')
                            else:
                                s_tmp = Condition(column=col_idx,min_value=float(value),type='continuous')
                            s.max = max(s.max,s_tmp.max)
                            s.min = min(s.min,s_tmp.min)
                        else:
                            # categorical
                            s.values.append(value)

                # merge, for example, merge "1<x<2" and "2<x<3" into "1<x<3"
                rule = Rule(conditions=conditions,domain=domain,target_class_idx=self.target_class_idx)
                rule_set.append(rule)
            return rule_set

        identified_rules = Orange_rules_to_our(Best_rules,self.domain,self.target_class)
        identified_rules = list(set(identified_rules))
        return identified_rules


    def generate_rule_space(self):
        """
        using fp-growth to generate the space of rules.
        note that it needs itemset, so the first step is to translate the data in the form of item set
        """

        supp = self.supp
        print("start Generating rule space.  Min support:",supp)

        # positive_indices = np.where(self.data_table.Y == self.target_class_idx)[0]
        # tmp_table = self.data_table[positive_indices]
        tmp_table = Table.from_numpy(X=self.total_X,Y=self.total_Y,domain=self.domain)
        # first discretize the data
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=5)
        # disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
        # disc.method = Orange.preprocess.discretize.EqualWidth(n=3)
        disc_data_table = disc(tmp_table)
        self.disc_data_table = disc_data_table

        X,mapping = OneHot.encode(disc_data_table,include_class=False)
        itemsets = list(frequent_itemsets(X,supp))
        decoder_item_to_feature = { idx:(feature.name,value) for idx,feature,value in OneHot.decode(mapping,disc_data_table,mapping)}

        self.rule_space = self.itemsets_to_rules(itemsets,self.domain,decoder_item_to_feature,self.target_class)
        self.rule_screening()
        self.rule_space = sorted(self.rule_space)
        print("number",len(self.rule_space))
        self.rule_space = sorted(set(self.rule_space))
        print("number",len(self.rule_space))
        # for r in self.rule_space:
        #     print(rule_to_string(r,self.domain,1))

    def itemsets_to_rules(self,itemsets,domain,decoder_item_to_feature,target_class):
        import re
        rule_set = []
        col_names = [it.name for it in domain.attributes]
        for itemset,_ in itemsets:
            previous_cols =[]
            conditions=[]
            for item in itemset:
                name,string_conditions = decoder_item_to_feature[item]
                col_idx = col_names.index(name)
                if col_idx not in previous_cols:
                    if domain.attributes[col_idx].is_continuous:
                        c = re.split( "- | ",string_conditions)
                        if len(c) == 2:
                            if c[0] in ['<=','≤','<']:
                                s = Condition(column=col_idx,max_value=float(c[1]),type='continuous')
                            else:
                                s = Condition(column=col_idx,min_value=float(c[1]),type='continuous')
                            conditions.append(s)
                        if len(c) == 3:
                            s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous')
                            conditions.append(s)
                    else:
                        # categorical
                        s = Condition(column=col_idx,values=[domain.attributes[col_idx].values.index(string_conditions)],type='categorical')
                        conditions.append(s)
                else:
                    s = conditions[ previous_cols.index(col_idx) ]
                    if domain.attributes[col_idx].is_continuous:
                        c = re.split( "- | ",string_conditions)
                        if len(c) == 2:
                            if c[0] in ['<=','≤','<']:
                                s_tmp = Condition(column=col_idx,max_value=float(c[1]),type='continuous')
                            else:
                                s_tmp = Condition(column=col_idx,min_value=float(c[1]),type='continuous')
                            s.max = max(s.max,s_tmp.max)
                            s.min = min(s.min,s_tmp.min)
                        if len(c) == 3:
                            s.max = max(s.max,float(c[2]))
                            s.min = min(s.min,float(c[0]))
                    else:
                        # categorical
                        s.values.append(domain.attributes[col_idx].values.index(string_conditions))
                previous_cols.append(col_idx)
            rule = Rule(conditions=conditions,domain=domain,target_class_idx=self.target_class_idx)
            rule_set.append(rule)
        return rule_set

    def rule_screening(self):
        ''' this screening process, the code is from the bayesian rule set paper. '''
        N_screening = 2000

        print ('Screening rules using information gain')
        def accumulate(iterable, func=operator.add):
            'Return running totals'
            # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
            # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
            it = iter(iterable)
            total = next(it)
            yield total
            for element in it:
                total = func(total, element)
                yield total
        from scipy.sparse import csc_matrix
        import time
        import itertools
        # self.supp = 50
        start_time = time.time()
        indices = np.array(list(itertools.chain.from_iterable([[ r.column for r in rule.conditions] for rule in self.rule_space])))
        len_rules = [len(rule.conditions) for rule in self.rule_space]
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(self.domain.attributes),len( self.rule_space )))
        # mat = np.matrix(df) * ruleMatrix
        mat = self.total_X * ruleMatrix
        Y_correct = np.equal(self.total_Y,self.target_class_idx)

        lenMatrix = np.matrix([len_rules for i in range(self.total_X.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)
        Zpos = [Z[i] for i in np.where(Y_correct>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        supp_select = np.where(TP>=self.supp*sum(Y_correct)/100)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        TN = len(Y_correct) - np.sum(Y_correct) - FP
        FN = np.sum(Y_correct) - TP
        p1 = TP.astype(float)/(TP+FP)
        p2 = FN.astype(float)/(FN+TN)
        pp = (TP+FP).astype(float)/(TP+FP+TN+FN)
        tpr = TP.astype(float)/(TP+FN)
        fpr = FP.astype(float)/(FP+TN)
        cond_entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
        cond_entropy[p1*(1-p1)==0] = -((1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2)))[p1*(1-p1)==0]
        cond_entropy[p2*(1-p2)==0] = -(pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1)))[p2*(1-p2)==0]
        cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
        select = np.argsort(cond_entropy[supp_select])[::-1][-N_screening:]
        self.rule_space = [self.rule_space[i] for i in supp_select[select]]
        # self.RMatrix = np.array(Z[:,supp_select[select]])
        print ('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(self.rule_space)) )

class Action():

    def __init__(self,current_solution,modification,X,Y,domain,current_obj,beta,lambda_parameter=0.01,target_class_idx=1,rho=None):
        mode = modification[0]
        self.mode = mode
        self.current_solution = current_solution
        self.current_obj = current_obj
        self.empirical_obj = current_obj
        self.hopeless = False
        # self.new_solution = [ Rule(conditions=[deepcopy(c) for c in r.conditions],domain=domain,target_class_idx=target_class_idx) for r in self.current_solution];

        # self.new_solution = deepcopy(self.current_solution);
        if mode == 'ADD_RULE':
            _,add_rule = modification
            # self.new_solution.append(add_rule)
            self.new_solution = self.current_solution[:] + [add_rule]
            self.changed_rule_idx = len(self.current_solution) # note that the index start from zero
        elif mode == 'REMOVE_RULE':
            _,remove_rule_idx = modification
            self.remove_rule = self.current_solution[remove_rule_idx]
            # del self.new_solution[remove_rule_idx]
            self.new_solution = [ c for idx,c in enumerate(self.current_solution) if idx!=remove_rule_idx]
            self.changed_rule_idx = remove_rule_idx
        elif mode in ['ADD_CONDITION','REMOVE_CONDITION','EXPAND_CONDITION','SPECIFY_CONDITION']:
            _,old_rule,old_rule_idx,new_rule = modification

            # self.new_solution.remove(old_rule)
            # self.new_solution.append(new_rule)
            self.new_solution = [ c if idx!=old_rule_idx else new_rule for idx,c in enumerate(self.current_solution)]
            self.changed_rule_idx=old_rule_idx
        else:
            raise ValueError("do not know this action mode:<",mode,">. Please check, or this mode is not implemented")

        # self.changed_rule = self.new_solution[self.changed_rule_idx]
        if self.mode in ['ADD_RULE','ADD_CONDITION','REMOVE_CONDITION','EXPAND_CONDITION','SPECIFY_CONDITION']:
            self.changed_rule = self.new_solution[self.changed_rule_idx]
        else:
            self.changed_rule = self.current_solution[self.changed_rule_idx]
        # from core import extend_rule
        # self.extended_changed_rule = extend_rule(self.changed_rule,domain=domain)
        self.extended_changed_rule = self.changed_rule

        self.string_representation = str(self.changed_rule_idx)+"."+self.changed_rule.string_representation

        self.total_volume = self.changed_rule.compute_volume(domain)

        self.beta = beta
        self.lambda_parameter = lambda_parameter
        self.rho = rho
        self.this_rho = 0 # initialize values

        self.update_objective(X,Y,domain,target_class_idx=target_class_idx)

    def make_hopeless(self):
        self.hopeless = True
        self.empirical_obj = -10000

    def obj_estimation(self):
        return self.empirical_obj - self.current_obj

    def update_current_obj(self,current_obj):
        self.current_obj = current_obj

    def update_objective(self,X,Y,domain,target_class_idx=1):
        '''
        update objective. and update the confidence interval
        '''
        self.empirical_obj = simple_objective(self.new_solution,X,Y,lambda_parameter=self.lambda_parameter,target_class_idx=target_class_idx)

        if self.beta == 0:
            self.beta_interval=0
            return

        curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
        # for r in self.new_solution:
        #     curr_covered_or_not |= r.evaluate_data(X)
        curr_covered_or_not |= self.changed_rule.evaluate_data(X)
        num = np.sum(curr_covered_or_not)
        # if num > 1000:
        #     from utils import rule_to_string
        #     print(rule_to_string(self.changed_rule,domain=domain,target_class_idx=1))

        self.num = num

        if num == 0:
            # if there is no samples in this rule, by no doubt it has a large uncertainty.
            self.beta_interval = 1
            return
        elif self.total_volume == 0:
            self.make_hopeless()
            self.beta_interval = 0
            return
        else:
            # this_rho = ( num / self.total_volume / ( rule_average_nearest_distance + 1e-10 )  ) + 1e-10
            this_rho = ( num / self.total_volume  ) + 1e-10
            self.this_rho = this_rho
            self.beta_interval =  self.beta *  math.sqrt (  self.rho / this_rho)
            # self.beta_interval =  self.beta *  (self.rho / this_rho)**2
            return
        return

    def __lt__(self,other):
        '''
        this less than comparision is crucial... make sure to implement it.
        As it will be used to select the best and second best action. as in the function in `core.py`
        '''
        return self.obj_estimation() < other.obj_estimation()

    def __hash__(self):
        '''
        this hash function is also important, it is used to avoid duplicated actions
        '''
        l = [self.mode,self.changed_rule_idx,hash(self.extended_changed_rule)  ]
        return hash(tuple(l))

    def __eq__(self,other):

        # return all([ self.mode == other.mode, self.changed_rule_idx == other.changed_rule_idx,self.changed_rule==other.changed_rule])
        return all([ self.mode == other.mode, self.changed_rule_idx == other.changed_rule_idx,self.extended_changed_rule==other.extended_changed_rule])

    def interval(self):
        return self.beta_interval

    def upper(self):
        return self.obj_estimation() + self.interval()

    def lower(self):
        return self.obj_estimation() - self.interval()
