import operator
import math
from functools import reduce
import numpy as np
import random
from copy import deepcopy,copy
import heapq

OPERATORS = {
    # discrete, nominal variables
    # 'in': lambda x,y: all( [operator.contains(y,x_tmp) for x_tmp in x ],
    'in': lambda x,y: operator.contains(y,x),
    # continuous variables
    '<=': operator.le,
    '>=': operator.ge,
}

def rule_to_string(rule,domain,target_class_idx):
    attributes = domain.attributes
    class_var = domain.class_var
    if rule.conditions:
        cond = " AND ".join([
                            str(s.min) + " <= " +
                            attributes[s.column].name +
                            " <= " + str(s.max)
                            if attributes[s.column].is_continuous
                            else attributes[s.column].name + " is " + ",".join( [ str(attributes[s.column].values[int(v)]) for v in s.values ])
                            for s in rule.conditions
                            ])
    else:
        cond = "TRUE"
    outcome = class_var.name + "=" + class_var.values[int(target_class_idx) ]
    return "IF {} THEN {} ".format(cond, outcome)

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
            if domain != None:

                max_limit = domain.attributes[column].max
                min_limit = domain.attributes[column].min
                min_value,max_value = sorted([min_value,max_value])
                self.min = max(min_value,min_limit) # for continuous features
                self.max = min(max_value,max_limit) # for continuous features
            else:
                self.min = min_value
                self.max = max_value
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

    def filter_data(self,X):
        return self.filter_data_(X)

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
            except:
                print("error in condition filter data")
                # print(self.values)
                # print(X[:5,self.column])
                # print(np.logical_or.reduce(
                # [np.equal(X[:5,self.column], v) for v in self.values ] ))
                quit()
        elif self.type == 'continuous':
            return np.logical_and(
            OPERATORS['<='](X[:,self.column], self.max),
            OPERATORS['>='](X[:,self.column], self.min)
            )

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

class Rule:
    def __init__(self,conditions,target_class_idx):
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
        # self.domain = domain
        # if string_representation == None:
        #     self.string_representation = rule_to_string(self,domain=domain,target_class_idx=self.target_class_idx)
        # else:
        #     self.string_representation = string_representation

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
        return self.evaluate_data_(X)

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
        try:
            if len(self.conditions) > 0:
                tmp = np.stack([condition.filter_data(X) for condition in self.conditions])
                # curr_covered = np.logical_and.reduce(tmp)
                curr_covered = np.all(tmp,axis=0)
            else:
                curr_covered = np.ones(X.shape[0], dtype=np.bool)
            # quit()
            return curr_covered
        except:
            # print(len(self.conditions))
            # print(self.string_representation)
            # print([len(c.values) for c in self.conditions])
            # print("*".join([str(c.values) for c in self.conditions]))
            # print([t.shape for t in tmp])
            print("error!")
            quit()

    def __len__(self):
        return len(self.conditions)

    def __hash__(self):
        return hash(self.string_representation)

    def __eq__(self, other):
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
        return Rule(conditions=[ deepcopy(c,memodict) for c in self.conditions],target_class_idx=self.target_class_idx)

    def add_constraint(self,feature,threshold,domain,mode="left_child"):
        if domain.attributes[feature].is_continuous:
            type = "continuous"
            if feature in [c.column for c in self.conditions]:
                condition = [ c for c in self.conditions if c.column==feature]
                assert len(condition)==1,"Strange thing! One condition should only appear one time"
                condition = condition[0]
                if mode == "left_child":
                    condition.max = threshold
                else:
                    condition.min = threshold
            else:
                if mode == "left_child":
                    new_condition = Condition(column=feature,max_value=threshold,domain=domain,type='continuous')
                else:
                    new_condition = Condition(column=feature,min_value=threshold,domain=domain,type='continuous')
                self.conditions.append(new_condition)
                self.conditions = sorted(self.conditions,key= lambda x:x.column)
        else:
            type = "categorical"
            if feature in [c.column for c in self.conditions]:
                condition = [ c for c in self.conditions if c.column==feature]
                assert len(condition)==1,"Strange thing! One condition should only appear one time"
                condition = condition[0]
                all_possoble_values = condition.values
                if mode == "left_child":
                    new_values = [ v for v in all_possoble_values if v <= threshold]
                    condition.values = new_values
                else:
                    new_values = [ v for v in all_possoble_values if v > threshold]
                    condition.values = new_values
            else:
                all_possoble_values = range(len(domain.attributes[feature].values))
                if mode == "left_child":
                    new_values = [ v for v in all_possoble_values if v <= threshold]
                    new_condition = Condition(column=feature,values=new_values,type='categorical')
                else:
                    new_values = [ v for v in all_possoble_values if v > threshold]
                    new_condition = Condition(column=feature,values=new_values,type='categorical')
                self.conditions.append(new_condition)

            self.conditions = sorted(self.conditions,key= lambda x:x.column)


    def set_target_class_idx(self,idx,domain):
        self.target_class_idx=idx
        self.string_representation = rule_to_string(self,domain=domain,target_class_idx=self.target_class_idx)
