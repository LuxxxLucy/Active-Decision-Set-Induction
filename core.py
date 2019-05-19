'''
    the core function of our approach:
    1. definition of data Structure
    2. essential functions used by pseudocode-like API
'''
import numpy as np
import math
import random
from collections import namedtuple
import operator
import copy

from Orange.data import Table, Domain

# hyper parameters:
# details of the meaning of hyperparameters, please see README
# the intervals to check when split the tree, for continuous features
K_INTERVALS = 100
# a hyper-parameter: the threshold of error
THRESHOLD = 0.01
# the threshold that this feature should not be split at all
SMALL_THRESHOLD = 0.01
# positive infinity
HUGE_NUMBER = 10000
# the minimum volume of a space
SMALL_SPACE =  10e-8
# minimum support of real instances for a pattern
MIN_SUPPORT =1


class Selector():
    OPERATORS = {
        # discrete, nominal variables
        # '==': operator.eq,
        # '=': operator.eq,
        # '!=': operator.ne,
        'in': lambda x,y: operator.contains(y,x),
        # note the reverse order of argument in the contains function

        # continuous variables
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
    }
    def __init__(self,column, values=None, min_value=None,max_value=None, type='categorical'):
        '''
        differs in the number of arguments as for categorical feature and continuous features
        '''
        self.type = type
        if type == 'categorical':
            # for categorical features
            self.column = column
            # value_1 is a set of possible values
            self.values = values
        elif type == 'continuous':
            # for continuous features
            self.column = column
            self.min = min_value
            self.max = max_value
        else:
            print("critical error. Type of selector unspecified. Must be one of categorical or continuous ")

    def filter_instance(self, x):
        """
        Filter a single instance. Returns true or false
        """
        if self.type == 'categorical':
            return Selector.OPERATORS['in'](x[self.column], self.values)
        elif self.type == 'continuous':
            return Selector.OPERATORS['<='](x[self.column], self.max) and Selector.OPERATORS['>='](x[self.column], self.min)

    def filter_data(self, X):
        """
        Filter several instances concurrently. Retunrs array of bools
        """

        if self.type == 'categorical':
            return Selector.OPERATORS['in'](X[:,self.column], self.values)
        elif self.type == 'continuous':
            return np.logical_and(
            Selector.OPERATORS['<='](X[:,self.column], self.max),
            Selector.OPERATORS['>='](X[:,self.column], self.min)
            )

    def sample(self,batch_size=10):
        if self.type == 'categorical':
            synthetic_column = np.random.choice(self.values,size=batch_size)
            print("all possible values",self.values)
            print("generated values",synthetic_instances)
            print("check this categorical generation!")
            quit()
            return synthetic_instances
        elif self.type == 'continuous':
            synthetic_column = np.random.uniform(low=self.min,high=self.max,size=batch_size)
            return synthetic_column

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



def diverse_filter_sorted(nodes,tree,dataset):

    # TODO: rank rules via diversity criteria

    # TODO: post-process: remove unnecessary limits, etc

    # TODO: replce simple implementation
    # nodes = sorted(nodes, key=lambda x: x.compute_space_coverage(),reverse=True)
    nodes = sorted(nodes, key=lambda x: x.q_value,reverse=True)
    return nodes

def compute_coverage(this_selectors,base_selectors):
    def compute_coverage_single(a,base):
        if a.type != base.type or a.column != base.column:
            raise Exception('critical error: unconsistent, a.type {0}, base.type{1}; a.column:{2},base.column{3}'.format(a.type,base.type,a.column,base.column))
        if a.type == 'categorical':
            return len(a.values) * 1.0 / len(base.values)
        elif a.type == 'continuous':
            return (a.max - a.min) * 1.0 / (base.max - base.min)
        else:
            raise Exception('critical error: unknown selector type a.type {0}, base.type'.format(a.type,base.type))

    space_coverage_values =  np.array( [ compute_coverage_single(this,base) for this,base in zip(this_selectors,base_selectors)   ] )
    space_coverage = np.prod(space_coverage_values)
    return space_coverage

class Pattern:
    """
    Represent a single pattern as a conjunction of many conditions and keep a reference to its parent.
    instance references are strictly not kept.

    Those can be easily gathered however, by following the trail of
    covered examples from rule to rule, provided that the original
    learning data reference is still known.
    """
    def __init__(self, selectors=None,target_class_idx=1):
        """
        Initialise a Rule.

        Parameters
        ----------
        selectors : list of condition selectors
        parent_rule : a reference to the parent rule.
        domain : Orange.data.domain.Domain
            Data domain, used to calculate class distributions.
        initial_class_dist : ndarray
            Data class distribution in regard to the whole learning set.
        prior_class_dist : ndarray
            Data class distribution just before a rule is developed.
        quality_evaluator : Evaluator
            Evaluation algorithm.
        complexity_evaluator : Evaluator
            Evaluation algorithm.
        significance_validator : Validator
            Validation algorithm.
        general_validator : Validator
            Validation algorithm.
        """
        self.selectors = selectors if selectors is not None else []
        self.target_class_idx =  target_class_idx

    def filter_and_store(self, X, Y, W, target_class, predef_covered=None):
        """
        Apply data and target class to a rule.

        Parameters
        ----------
        X, Y, W : ndarray
            Learning data.
        target_class : int
            Index of the class to model.
        predef_covered : ndarray
            Built-in optimisation variable to enable external
            computation of covered examples.
        """
        # self.target_class = target_class
        # if predef_covered is not None:
        #     self.covered_examples = predef_covered
        # else:
        #     self.covered_examples = np.ones(X.shape[0], dtype=bool)
        #     for selector in self.selectors:
        #         self.covered_examples &= selector.filter_data(X)
        #
        # self.curr_class_dist = get_dist(Y[self.covered_examples],
        #                                 W[self.covered_examples]
        #                                 if W is not None else None,
        #                                 self.domain)
        pass

    def is_valid(self):
        """
        Return True if the rule passes the general validator's
        requirements.
        """
        # return self.general_validator.validate_rule(self)
        pass

    def evaluate_instance(self, x):
        """
        Evaluate a single instance.

        Parameters
        ----------
        x : ndarray
            Evaluate this instance.

        Returns
        -------
        res : bool
            True, if the rule covers 'x'.
        """
        return all(selector.filter_instance(x) for selector in self.selectors)

    def evaluate_data(self, X):
        """
        Evaluate several instances concurrently.

        Parameters
        ----------
        X : ndarray
            Evaluate this data.

        Returns
        -------
        res : ndarray, bool
            Array of evaluations.
        """
        curr_covered = np.ones(X.shape[0], dtype=bool)
        for selector in self.selectors:
            curr_covered &= selector.filter_data(X)
        return curr_covered

    def __eq__(self, other):
        # return self.selectors == other.selectors
        return np.array_equal(self.covered_examples, other.covered_examples)

    def __len__(self):
        return len(self.selectors)

    def string_representation(self,domain):
        attributes = domain.attributes
        class_var = domain.class_var
        if self.selectors:
            cond = " AND ".join([
                                str(s.min) + " <= " +
                                attributes[s.column].name +
                                " <= " + str(s.max)
                                if attributes[s.column].is_continuous
                                else attributes[s.column].name + " in " + str(attributes[s.column].values[int(s.values)])
                                for s in self.selectors
                                ])
        else:
            cond = "TRUE"
        outcome = class_var.name + "=" + class_var.values[self.target_class_idx]
        return "IF {} THEN {} ".format(cond, outcome)



class Node:

    def __init__(self,domain,target_class,selectors,parent,real_instances=None, synthetic_instances=None,base_selectors=None):
        '''
        A node in the MCTS tree.
        Basically, there are three types of variables
        1. variables that is relaetd to the tree structure. such as reference to parent and child nodes.
        2. `static` variables that won't be modified after initialization. such as target class, etc
        3. `dynamic` variables that will be modified on-line. such as q-value, count numebers...
        '''

        # variables about the tree structure
        self.parent = parent
        self.children = []

        # static variables of this node that will not be modified after initialization
        # information about current_node
        self.domain = domain
        self.target_class = target_class
        target_class_idx = domain.class_var.values.index(target_class)
        self.pattern = Pattern(selectors=selectors, target_class_idx=target_class_idx)
        self.base_selectors = base_selectors
        if base_selectors == None:
            raise Exception('error! base_selectors wrong!')


        # dynamic variables of this node that will be modified
        self.real_instances = real_instances # TODO and disclaimer: it make sense check that instances are really covered by the pattern. But due to duplicate computational cost, I do want to do it for now.
        self.synthetic_instances = synthetic_instances if synthetic_instances != None else Table.from_domain(domain=domain,n_rows=0)
        self.count = len(self.real_instances) + len(self.synthetic_instances)
        self.positive = 0
        self.true_positive = 0
        self.precision = 0
        self.space_coverage = 0
        self.q_value = self.compute_q()
        self.hopeless = False

        # indication of feature being modifed or not
        if parent != None:
            self.dirty_indication = parent.dirty_indication
            if np.mean(self.dirty_indication) != 1:
                col_different = np.array(  [ a == b for a,b in zip(parent.pattern.selectors,self.pattern.selectors) ] )
                col_idx = np.where( col_different == False ) [0][0]
                self.dirty_indication[col_idx] = True
            else:
                # it means that all attributes are dirty, no needs for further computation
                pass
        else:
            self.dirty_indication = np.zeros( len(domain.attributes) ,dtype=bool)


    def is_terminal(self):
        '''
        either if this node has no children
        or this node has be "finished" satisfying some criteria
        '''
        return self.children == []

    def is_satisfied(self):
        return self.precision >= 1- THRESHOLD

    def is_hopeless(self):
        # TODO: rethink here, is min_support is necessary
        # if self.is_terminal():
        #     # if there is no real instance, there will be no positive real instances as support, this is an immediate "hopeless"
        #     # note that in the tree, non-leaf node do not contain
        #     return np.sum(self.real_instances.Y == 1 ) < MIN_SUPPORT or self.precision <= THRESHOLD or self.hopeless == True or self.space_coverage <= SMALL_SPACE
        return self.precision <= THRESHOLD or self.hopeless == True or self.space_coverage <= SMALL_SPACE


    def is_finished(self):
        '''
        A simple threshold based criteria.
        'finished' means this node should not be considered in the Tree Selection phase for criteria like UCT
        This means either a pattern is okay (is_satisfied == True) with a high precision or is totally hopeless
        '''
        return self.is_satisfied() or self.is_hopeless()

    def is_significant(self):
        return np.sum(self.real_instances.Y == 1 ) >= MIN_SUPPORT and self.space_coverage >= SMALL_SPACE
    def split_or_not(self):
        # TODO: change split or not
        # return is_satisfied() or is_hopeless()
        return True

    def compute_q(self):
        '''
        compute q value based on real instances and current synthetic_instances
        also update `true_positive` and `positive`
        '''
        tmp_instances_set = Table.from_table(self.domain,self.real_instances)
        tmp_instances_set.extend(self.synthetic_instances)
        if len(tmp_instances_set) ==0:
            return 0
        X,Y = tmp_instances_set.X, tmp_instances_set.Y
        size = X.shape[0]
        covered_instances = np.ones(size,dtype=bool)
        for selector in self.pattern.selectors:
            covered_instances &= selector.filter_data(X)
        target_covered_instances = np.logical_and(  covered_instances, Y==1)

        self.true_positive = np.sum(target_covered_instances) # true_positive means the number of instancs that covered by the pattern and is also in the target class labeled by the blackbox
        self.positive = np.sum(covered_instances) # positive means the number of instances that are covered by this pattern
        self.precision = self.true_positive / self.positive
        self.space_coverage = self.compute_space_coverage()

        return self.precision * self.space_coverage

    def update_current(self,new_synthetic_instances):
        '''
        update q, true_positive,positive.
        also append new samples into self.synthetic_instances
        '''

        X,Y = new_synthetic_instances.X, new_synthetic_instances.Y
        size = X.shape[0]
        covered_instances = np.ones(size,dtype=bool)
        for selector in self.pattern.selectors:
            covered_instances &= selector.filter_data(X)
        target_covered_instances = np.logical_and(  covered_instances, Y==1)
        new_true_positive = np.sum(target_covered_instances)
        new_positive = np.sum(covered_instances)
        # update some key variables

        self.positive += new_positive
        self.true_positive += new_true_positive
        self.precision = self.true_positive / self.positive
        self.space_coverage = self.compute_space_coverage()
        self.q_value = self.precision * self.space_coverage

        # append these new instances
        self.synthetic_instances.extend(new_synthetic_instances)
        self.count += len(new_synthetic_instances)

    def compute_space_coverage(self):
        space_coverage = compute_coverage(self.pattern.selectors,self.base_selectors)
        return space_coverage

    def split_info_gain(self):
        '''
        split useing information gain criteria to find a single feature and threshold to split.
        Note that we use value set split for categorical features.
        '''
        def info_entropy(p,n):
            # Calculate entropy
            if p  == 0 or n == 0:
            	I = 0
            else:
            	I = ((-1*p)/(p + n))* math.log(p/(p+n), 2) + ((-1*n)/(p + n))* math.log(n/(p+n), 2)
            return I

        def info_gain(s,p_1,p_2,instances):
            '''
            calculate the information gain
            '''
            all_instances = np.ones(instances.X.shape[0],dtype=bool)
            # s_positive = np.sum( s.filter_data(instances.X)   )
            # s_negative = instance.X.shape[0] - s_positive
            s_positive = np.sum(instances.Y == 1)
            s_negative = np.sum(instances.Y == 0)
            s_all = len(instances)

            index = np.where( p_1.evaluate_data(instances.X) )[0]
            instances_s_1 = Table.from_table(instances.domain,instances,row_indices=index)
            s_1_positive = np.sum(instances_s_1.Y == 1)
            s_1_negative = np.sum(instances_s_1.Y == 0)
            s_1_all = len(instances_s_1)

            index = np.where( p_2.evaluate_data(instances.X) )[0]
            instances_s_2 = Table.from_table(instances.domain,instances,row_indices=index)
            s_2_positive = np.sum(instances_s_2.Y == 1)
            s_2_negative = np.sum(instances_s_2.Y == 0)
            s_2_all = len(instances_s_2)

            info_gain = info_entropy(s_positive,s_negative) - s_1_all / (s_all+1) * info_entropy(s_1_positive,s_2_negative) - s_2_all / (s_all+1) * info_entropy(s_2_positive,s_2_negative)
            return info_gain

        def best_split_single_feature(i,pattern,real_instances,synthetic_instances):
            '''
            return the best split info gain for a given feature (a given selector)
            '''
            instances = Table.from_table(real_instances.domain,real_instances)
            selectors = pattern.selectors
            selector = selectors[i]
            if selector.type == 'continuous':
                def split_selector(s,threshold):
                    s_1 = copy.deepcopy(selectors); s_1[i].max = threshold
                    s_2 = copy.deepcopy(selectors); s_2[i].min = threshold
                    p_1 = Pattern(selectors=s_1,target_class_idx=pattern.target_class_idx)
                    p_2 = Pattern(selectors=s_2,target_class_idx=pattern.target_class_idx)
                    return (p_1,p_2)
                # TODO: make this small threshold to percentage
                if (selector.max - selector.min) <= SMALL_THRESHOLD:
                    return (None,None,-HUGE_NUMBER)
                option_thresholds = np.linspace(selector.min,selector.max,num=K_INTERVALS)[1:-1]
                options = [ (option_threshold,split_selector(selector,option_threshold)) for option_threshold in option_thresholds]
            elif selector.type == 'categorical':
                print('best_split_single_feature not implemented for categorical feature')
                quit()
            else:
                print('unknown error')
                quit()

            tuples = []
            for option,(p_1,p_2) in options:
                # construct new selector for a given threshold option
                value = info_gain(selectors,p_1,p_2,instances)
                tuples.append( ( option,p_1,p_2,value  ))

            _,p_1,p_2,value = max(tuples, key=lambda x: x[3])

            return (p_1,p_2,value)

        def select_best_split(pattern,real_instances,synthetic_instances):
            '''
            return best two new selectors after split
            '''
            selectors = pattern.selectors
            pairs = [ (s.column, best_split_single_feature(i,pattern,real_instances,synthetic_instances) ) for i,s in enumerate(selectors)  ]
            # argmax, finding the most info_gain
            col, (p_1,p_2,_) = max(pairs, key=lambda x: x[1][2])
            return col,p_1, p_2

        col, p_1,p_2 = select_best_split(self.pattern,self.real_instances,self.synthetic_instances)
        # form new nodes
        # needs to transfer samples into leaf nodes (note that parent node will not contain instances anymore.)

        if p_1 is None and p_2 is None:
            # this means that this split is too small according to the threshold
            self.hopeless = True
            return False

        index = np.where( p_1.evaluate_data(self.real_instances.X) )[0]
        real_instances_1 = Table.from_table(self.real_instances.domain,self.real_instances,row_indices=index)
        index = np.where( p_1.evaluate_data(self.synthetic_instances.X) )[0]
        synthetic_instances_1 = Table.from_table(self.synthetic_instances.domain,self.synthetic_instances,row_indices=index)
        child_1 =  Node(self.domain,self.target_class, selectors=p_1.selectors, parent=self, real_instances=real_instances_1,synthetic_instances=synthetic_instances_1,base_selectors=self.base_selectors)

        index = np.where( p_2.evaluate_data(self.real_instances.X) )[0]
        real_instances_2 = Table.from_table(self.real_instances.domain,self.real_instances,row_indices=index)
        index = np.where( p_2.evaluate_data(self.synthetic_instances.X) )[0]
        synthetic_instances_2 = Table.from_table(self.synthetic_instances.domain,self.synthetic_instances,row_indices=index)
        child_2 =  Node(self.domain,self.target_class, selectors=p_2.selectors, parent=self, real_instances=real_instances_2,synthetic_instances=synthetic_instances_2,base_selectors=self.base_selectors)

        self.children.extend([child_1,child_2])
        self.real_instances=[]
        self.synthetic_instances=[]
        return True

    def split_WRAcc(self):
        '''
        split using weighted relative accuracy criteria to find a single feature and threshold to split.
        Note that we use value set split for categorical features.
        '''
        def WRAcc(s,p_1,p_2,instances):
            '''
            calculate the information gain
            '''
            all_instances = np.ones(instances.X.shape[0],dtype=bool)
            # s_positive = np.sum( s.filter_data(instances.X)   )
            # s_negative = instance.X.shape[0] - s_positive
            s_positive = np.sum(instances.Y == 1)
            s_negative = np.sum(instances.Y == 0)
            s_all = len(instances)

            index = np.where( p_1.evaluate_data(instances.X) )[0]
            instances_s_1 = Table.from_table(instances.domain,instances,row_indices=index)
            s_1_positive = np.sum(instances_s_1.Y == 1)
            s_1_negative = np.sum(instances_s_1.Y == 0)
            s_1_all = len(instances_s_1)

            index = np.where( p_2.evaluate_data(instances.X) )[0]
            instances_s_2 = Table.from_table(instances.domain,instances,row_indices=index)
            s_2_positive = np.sum(instances_s_2.Y == 1)
            s_2_negative = np.sum(instances_s_2.Y == 0)
            s_2_all = len(instances_s_2)

            # info_gain = info_entropy(s_positive,s_negative) - s_1_all / (s_all+1) * info_entropy(s_1_positive,s_2_negative) - s_2_all / (s_all+1) * info_entropy(s_2_positive,s_2_negative)
            wracc_s_p1 = compute_coverage(p_1.selectors,s) * ( s_1_positive / (s_1_all+1)  - s_positive / (s_all+1)  )

            wracc_s_p2 = compute_coverage(p_2.selectors,s) * ( s_2_positive / (s_2_all+1) -   s_positive / (s_all+1)   )

            WRAcc_gain = max(wracc_s_p1,wracc_s_p2)

            return WRAcc_gain

        def best_split_single_feature(i,pattern,real_instances,synthetic_instances):
            '''
            return the best split info gain for a given feature (a given selector)
            '''
            instances = Table.from_table(real_instances.domain,real_instances)
            selectors = pattern.selectors
            selector = selectors[i]
            if selector.type == 'continuous':
                def split_selector(s,threshold):
                    s_1 = copy.deepcopy(selectors); s_1[i].max = threshold
                    s_2 = copy.deepcopy(selectors); s_2[i].min = threshold
                    p_1 = Pattern(selectors=s_1,target_class_idx=pattern.target_class_idx)
                    p_2 = Pattern(selectors=s_2,target_class_idx=pattern.target_class_idx)
                    return (p_1,p_2)
                # TODO: make this small threshold to percentage
                if (selector.max - selector.min) <= SMALL_THRESHOLD:
                    return (None,None,-HUGE_NUMBER)
                option_thresholds = np.linspace(selector.min,selector.max,num=K_INTERVALS)[1:-1]
                options = [ (option_threshold,split_selector(selector,option_threshold)) for option_threshold in option_thresholds]
            elif selector.type == 'categorical':
                print('best_split_single_feature not implemented for categorical feature')
                quit()
            else:
                print('unknown error')
                quit()

            tuples = []
            for option,(p_1,p_2) in options:
                # construct new selector for a given threshold option
                value = WRAcc(selectors,p_1,p_2,instances)
                tuples.append( ( option,p_1,p_2,value  ))

            _,p_1,p_2,value = max(tuples, key=lambda x: x[3])

            return (p_1,p_2,value)

        def select_best_split(pattern,real_instances,synthetic_instances):
            '''
            return best two new selectors after split
            '''
            selectors = pattern.selectors
            pairs = [ (s.column, best_split_single_feature(i,pattern,real_instances,synthetic_instances) ) for i,s in enumerate(selectors)  ]
            # argmax, finding the most info_gain
            col, (p_1,p_2,_) = max(pairs, key=lambda x: x[1][2])
            return col,p_1, p_2

        col, p_1,p_2 = select_best_split(self.pattern,self.real_instances,self.synthetic_instances)
        # form new nodes
        # needs to transfer samples into leaf nodes (note that parent node will not contain instances anymore.)

        if p_1 is None and p_2 is None:
            # this means that this split is too small according to the threshold
            self.hopeless = True
            return False

        index = np.where( p_1.evaluate_data(self.real_instances.X) )[0]
        real_instances_1 = Table.from_table(self.real_instances.domain,self.real_instances,row_indices=index)
        index = np.where( p_1.evaluate_data(self.synthetic_instances.X) )[0]
        synthetic_instances_1 = Table.from_table(self.synthetic_instances.domain,self.synthetic_instances,row_indices=index)
        child_1 =  Node(self.domain,self.target_class, selectors=p_1.selectors, parent=self, real_instances=real_instances_1,synthetic_instances=synthetic_instances_1,base_selectors=self.base_selectors)

        index = np.where( p_2.evaluate_data(self.real_instances.X) )[0]
        real_instances_2 = Table.from_table(self.real_instances.domain,self.real_instances,row_indices=index)
        index = np.where( p_2.evaluate_data(self.synthetic_instances.X) )[0]
        synthetic_instances_2 = Table.from_table(self.synthetic_instances.domain,self.synthetic_instances,row_indices=index)
        child_2 =  Node(self.domain,self.target_class, selectors=p_2.selectors, parent=self, real_instances=real_instances_2,synthetic_instances=synthetic_instances_2,base_selectors=self.base_selectors)

        self.children.extend([child_1,child_2])
        self.real_instances=[]
        self.synthetic_instances=[]
        return True

    def sample(self,batch_size=10):
        # sort with column index, this is to be consistent with table domain
        tmp_selectors = sorted(self.pattern.selectors,key=lambda x:x.column )
        raw_X_columns = [ s.sample(batch_size=batch_size) for s in tmp_selectors]
        raw_X = np.column_stack(raw_X_columns)
        return raw_X


    def __str__(self):
        return self.pattern.string_representation(self.domain)

    def summary(self):

        print("pattern:",self)
        print("precision:",self.precision)
        print("q_value:",self.q_value)
        print("count:",self.count)
        print("true positive:",self.true_positive)
        print("positive (instances covered by this rule):",self.positive)
