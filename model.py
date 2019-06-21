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
from core import simple_objective,get_correct_cover_ruleset,get_symmetric_difference,sample_new_instances
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

        X,Y = self.data_table.X,self.data_table.Y
        X_prime,Y_prime = self.synthetic_data_table.X,self.synthetic_data_table.Y
        self.total_X = np.append(X, X_prime, axis=0)
        self.total_Y = np.append(Y, Y_prime, axis=0)

        return

    def set_parameters(self,beta):
        # self.N_iter_max = 10
        self.N_iter_max = 1000
        self.N_batch = 10
        self.beta = beta
        self.epsilon = 0.01


        self.supp = 50
        self.maxlen=3
        print("target class is:",self.target_class,". Its index is",self.target_class_idx)
        # TODO: add hyperparameter print
        return


    def initialize(self):
        self.termination = False
        # self.current_solution = random.sample(self.rule_space,3)
        self.current_solution = []
        self.current_obj = simple_objective(self.current_solution,self.data_table.X,self.data_table.Y,target_class_idx=self.target_class_idx)
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
            # todo: add obj update in the tqdm log
            # print("new best obj:",self.best_obj)
        return

    def update_current_solution(self,best_action,actions):

        t = random.random()
        if t < self.epsilon:
            a = random.choice(actions)
        else:
            a = best_action
        self.current_solution = a.new_solution
        return


    def generate_action_space(self):

        actions = self.generate_action(self.total_X,self.total_Y,beta=self.beta)
        return actions

    def generate_synthetic_instances(self,a_star,a_prime):
        region_1,region_2 = get_symmetric_difference(a_star,a_prime,self.domain)
        X_new = sample_new_instances(region_1,region_2,self.total_X,self.total_Y,self.domain)

        return X_new

    def update_actions(self,actions,a_star,a_prime,X_new):
        labels = self.blackbox(self.encoder(X_new))
        XY_new = Orange.data.Table(self.domain, X_new, labels)
        self.synthetic_data_table.extend(XY_new)
        self.total_X = np.append(self.total_X, XY_new.X, axis=0)
        self.total_Y = np.append(self.total_Y, XY_new.Y, axis=0)

        for a in actions:
            a.update_objective(self.total_X,self.total_Y,self.domain,target_class_idx=self.target_class_idx)
        return actions

    def update(self):
        self.count+=1

    def output(self):
        return self.best_solution

    def compute_accuracy(self):
        theta = len(get_correct_cover_ruleset(self.current_solution,self.data_table.X,self.data_table.Y,target_class_idx=self.target_class_idx)) * 1.0 / len(self.data_table.X)
        return theta
