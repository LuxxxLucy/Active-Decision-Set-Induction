# import some critical packages we use.
import numpy as np
import Orange

# import standard programming-purpose package
import time
from copy import deepcopy
import random

# import math

# import local package
from utils import rule_to_string
from core import simple_objective,get_incorrect_cover_ruleset
from structure import DecisionSet

class ADS(DecisionSet):
    '''
    The Active Decision Set model
    '''
    def __init__(self,data_table, blackbox,encoder,target_class='yes',seed=42):
        random.seed(seed)
        self.count=0

        self.data_table = data_table
        self.domain = data_table.domain
        # for continuous variabels, we compute max and min
        for feature in self.domain.attributes:
            if feature.is_continuous:
                feature.max = max([ d[feature] for d in self.data_table ]).value
                feature.min = min([ d[feature] for d in self.data_table ]).value


        self.blackbox = blackbox
        self.encoder = encoder

        self.target_class = target_class
        # for example target_class = 'yes' and the domain Y is ['no','yes']
        # then the target class idx = 1
        self.target_class_idx = self.data_table.domain.class_var.values.index(self.target_class)

    def initialize_synthetic_dataset(self):
        self.synthetic_data_table = Orange.data.Table.from_domain(domain=self.domain,n_rows=0) # synthetic data (now empty)
        return

    def set_parameters(self):
        self.N_iter_max = 1000
        self.N_batch = 10
        self.beta = 10
        self.epsilon = 0.01
        return


    def initialize(self):
        self.termination = False
        # self.current_solution = random.sample(self.rule_space,3)
        self.current_solution = []
        self.current_obj = simple_objective(self.current_solution,self.data_table.X,self.data_table.Y)
        self.best_obj = self.current_obj
        self.best_solution = self.current_solution

    def termination_condition(self):
        if self.termination == True:
            return True
        if self.count >= self.N_iter_max:
            return True
        else:
            return False

    def update_best_solution(self,best_action):
        if self.best_obj < best_action.obj_estimation():
            self.best_obj = best_action.obj_estimation()
            self.best_solution = best_action.new_solution
            print("new best obj:",self.best_obj)
        return

    def update_current_solution(self,best_action,actions):

        t = random.random()
        if t < self.epsilon:
            a = random.choice(actions)
        else:
            a = best_action
        self.current_solution = a.new_solution
        return

    def update(self):
        self.count+=1

    def generate_action_space(self):
        X,Y = self.data_table.X,self.data_table.Y
        X_prime,Y_prime = self.synthetic_data_table.X,self.synthetic_data_table.Y
        X = np.append(X, X_prime, axis=0)
        Y = np.append(Y, Y_prime, axis=0)
        actions = self.generate_action(X,Y)
        # self.solution_neighbors = [ random.sample(self.rule_space,3) for _ in range(10)]
        # best_neighbor = max(self.solution_neighbors, key= lambda x: objective(x,self.rule_space,self.data_table.X,self.data_table.Y,self.lambda_array) )
        # current_obj = objective(best_neighbor,self.rule_space,self.data_table.X,self.data_table.Y,self.lambda_array)
        # if current_obj >= self.best_obj:
        #     self.best_obj = current_obj
        #     self.best_solution = best_neighbor
        return actions

    def output(self):
        return self.best_solution

    def compute_accuracy(self):
        theta = len(get_incorrect_cover_ruleset(self.current_solution,self.data_table.X,self.data_table.Y)) * 1.0 / len(self.data_table.X)
        return theta
