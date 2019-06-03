# import some critical packages we use.
import numpy as np
import Orange
from Orange.data import Table
from orangecontrib.associate.fpgrowth import frequent_itemsets, OneHot

# import standard programming-purpose package
import time
from copy import deepcopy
import random

# import math

# import local package
from utils import itemsets_to_rules,rule_to_string
from objective import objective

class ADS(object):
    '''
    The Active Decision Set model
    '''
    def __init__(self,data_table, blackbox,encoder,target_class='yes',seed=42):
        random.seed(seed)
        self.count=0
        self.Niter = 100

        self.data_table = data_table
        self.domain = data_table.domain
        # for continuous variabels, we compute max and min
        for feature in self.domain.attributes:
            if feature.is_continuous:
                feature.max = max([ d[feature] for d in self.data_table ])
                feature.min = min([ d[feature] for d in self.data_table ])

        self.blackbox = blackbox
        self.encoder = encoder

        self.target_class = target_class
        # for example target_class = 'yes' and the domain Y is ['no','yes']
        # then the target class idx = 1
        self.target_class_idx = self.data_table.domain.class_var.values.index(self.target_class)

        self.synthetic_data_table = Table.from_domain(domain=data_table.domain,n_rows=0) # synthetic data (now empty)


    def set_parameters(self):
        pass

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
        # disc.method = Orange.preprocess.discretize.EqualFreq(n=2)
        # disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
        disc.method = Orange.preprocess.discretize.EqualWidth(n=4)
        disc_data_table = disc(self.data_table[positive_indices])

        X,mapping = OneHot.encode(disc_data_table,include_class=False)
        itemsets = list(frequent_itemsets(X,self.supp))
        decoder_item_to_feature = { idx:(feature.name,value) for idx,feature,value in OneHot.decode(mapping,disc_data_table,mapping)}

        self.rule_space = itemsets_to_rules(itemsets,self.domain,decoder_item_to_feature,self.target_class)

    def initialize(self):
        self.termination = False
        self.solution = random.sample(self.rule_space,3)
        self.lambda_array = [1.0] * 7
        self.best_obj = objective(self.solution,self.rule_space,self.data_table.X,self.data_table.Y,self.lambda_array)
        self.best_solution = self.solution

    def termination_condition(self):
        if self.termination == True:
            return True
        if self.count >= self.Niter:
            return True
        else:
            return False

    def update(self):
        self.count+=1

    def generate_action_space(self):
        self.solution_neighbors = [ random.sample(self.rule_space,3) for _ in range(10)]
        best_neighbor = max(self.solution_neighbors, key= lambda x: objective(x,self.rule_space,self.data_table.X,self.data_table.Y,self.lambda_array) )
        current_obj = objective(best_neighbor,self.rule_space,self.data_table.X,self.data_table.Y,self.lambda_array)
        if current_obj >= self.best_obj:
            self.best_obj = current_obj
            self.best_solution = best_neighbor
        return

    def act(self,actions):
        pass

    def output(self):
        return self.best_solution
