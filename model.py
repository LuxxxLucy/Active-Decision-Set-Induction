# import some critical packages we use.
import numpy as np
import Orange
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,Normalizer
from sklearn.metrics.pairwise import euclidean_distances as distance_function

# import standard programming-purpose package
import time
from copy import deepcopy
import random

# import debug-purpose package
import logging

# import math
import math

# import local package
from utils import rule_to_string
from core import simple_objective,get_correct_cover_ruleset,get_symmetric_difference,sample_new_instances
from structure import DecisionSet


class ADS(DecisionSet):
    '''
    The Active Decision Set model
    '''
    def __init__(self,data_table, blackbox,target_class='yes',seed=42):
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
        self.target_class = target_class
        # for example target_class = 'yes' and the domain Y is ['no','yes']
        # then the target class idx = 1
        self.target_class_idx = self.data_table.domain.class_var.values.index(self.target_class)

        categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]
        continuous_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_continuous]

        self.preprocessor = make_column_transformer( ( OneHotEncoder(categories='auto',sparse=False),categorical_features_idx),
        (StandardScaler(), continuous_features_idx),
                            remainder = 'passthrough'
                            )
        self.preprocessor.fit(data_table.X)
        # self.transformer = lambda x : self.preprocessor.transform(x).toarray()
        self.transformer = lambda x : self.preprocessor.transform(x)

        self.solution_history=[]

    def initialize_synthetic_dataset(self):
        self.synthetic_data_table = Orange.data.Table.from_domain(domain=self.domain,n_rows=0) # synthetic data (now empty)
        X,Y = self.data_table.X,self.data_table.Y
        X_prime,Y_prime = self.synthetic_data_table.X,self.synthetic_data_table.Y
        self.total_X = np.append(X, X_prime, axis=0)
        self.total_Y = np.append(Y, Y_prime, axis=0)
        return

    def set_parameters(self,beta,lambda_parameter):
        self.N_iter_max = 1000
        # self.N_iter_max = 100
        self.lambda_parameter = lambda_parameter
        self.beta = beta

        # self.T_0 = 1000

        self.N_batch = 10
        self.epsilon = 0.01
        self.supp=0.05

        # print("target class is:",self.target_class,". Its index is",self.target_class_idx)
        # TODO: add hyperparameter print
        return

    def initialize(self):
        self.termination = False

        self.current_solution = []
        self.current_obj = simple_objective(self.current_solution,self.data_table.X,self.data_table.Y,lambda_parameter=self.lambda_parameter,target_class_idx=self.target_class_idx)
        # print("initial obj: ",self.current_obj)
        self.best_obj = deepcopy(self.current_obj)
        self.best_solution = deepcopy(self.current_solution)
        if hasattr(self,'rule_space'):
            delattr(self, 'rule_space')

        # print(" compute initial rho")
        self.rho = (self.data_table.X.shape[0] / 1 )
        # print(" compute initial rho")
        logging.info("initialization okay")


    def termination_condition(self):
        if self.termination == True:
            return True
        if self.count >= self.N_iter_max:
            return True
        else:
            return False

    def update_best_solution(self,best_action):
        self.best_obj = simple_objective(self.best_solution,self.total_X,self.total_Y,lambda_parameter=self.lambda_parameter,target_class_idx=self.target_class_idx)

        if self.best_obj < best_action.obj_estimation():
            # print("best obj",self.best_obj," -> new obj",best_action.obj_estimation(),"number of rules",len(best_action.new_solution))
            self.best_obj = best_action.empirical_obj
            self.best_solution = deepcopy(best_action.new_solution)
            # todo: add obj update in the tqdm log
            # print("new best obj:",self.best_obj,"number of rules",len(self.best_solution))
        return

    def update_current_solution(self,best_action,actions):

        t = random.random()

        if t < 0.001:
            '''re-start'''
            # print("re-staring!!!!!!!!!!!!")
            self.current_solution = []
            self.current_obj = simple_objective(self.current_solution,self.data_table.X,self.data_table.Y,lambda_parameter=self.lambda_parameter,target_class_idx=self.target_class_idx)
            if hasattr(self,'rule_space'):
                delattr(self, 'rule_space')
            return
        elif t < self.epsilon:
            # print("take random")
            a = random.choice(actions)
        else:
            a = best_action

        # T = self.T_0**(1 - self.count/self.N_iter_max)
        # prob = min(1, np.exp( (a.obj_estimation()- self.current_obj) / T)  )
        # print(prob)
        # if np.random.random() <= prob:
        #     self.current_solution = a.new_solution
        self.current_solution = deepcopy(a.new_solution)
        # todo: better solution than this, why could this happen?
        self.current_solution = list(set(self.current_solution))
        # self.current_obj = deepcopy(a.empirical_obj)
        self.current_obj = a.obj_estimation()

        return

    def generate_action_space(self):
        actions = self.generate_action(self.total_X,self.total_Y,beta=self.beta,rho=self.rho,transformer=self.transformer)

        # print(len(actions))
        # from utils import rule_to_string
        # for a in actions:
        #     for r in a.new_solution:
        #         print(rule_to_string(r,domain=dataset.domain,target_class_idx=1))
        #     print("---")
        #
        # quit()

        return actions

    def generate_synthetic_instances(self,a_star,a_prime):
        region_1,region_2 = get_symmetric_difference(a_star,a_prime,self.domain)
        # print(a_star.mode,a_prime.mode)
        X_new,Y_new = sample_new_instances(region_1,region_2,self.total_X,self.total_Y,self.domain,self.blackbox,transformer=self.transformer)
        return X_new,Y_new

    def generate_synthetic_instances_for_solution(self,sol_1,sol_2):
        # region_1,region_2 = get_symmetric_difference(a_star,a_prime,self.domain)
        X_new = sample_new_instances_for_solution(sol_1,sol_2,self.total_X,self.total_Y,self.domain,self.blackbox)
        return X_new

    def update_actions(self,actions,a_star,a_prime,X_new,Y_new):
        XY_new = Orange.data.Table(self.domain, X_new, Y_new)
        # curr_covered_or_not = np.zeros(XY_new.X.shape[0], dtype=np.bool)
        # curr_covered_or_not |= a_star.changed_rule.evaluate_data(XY_new.X)
        # print(np.sum(curr_covered_or_not))
        #
        # curr_covered_or_not = np.zeros(XY_new.X.shape[0], dtype=np.bool)
        # # for r in self.new_solution:
        # #     curr_covered_or_not |= r.evaluate_data(X)
        # curr_covered_or_not |= a_prime.changed_rule.evaluate_data(XY_new.X)
        # print(np.sum(curr_covered_or_not))

        self.synthetic_data_table.extend(XY_new)

        self.total_X = np.append(self.total_X, XY_new.X, axis=0)
        self.total_Y = np.append(self.total_Y, XY_new.Y, axis=0)

        start_time = time.time()
        # end_time = time.time()
        # print ('\tTook %0.3fs to generate the KDtree' % (time.time() - start_time ) )

        for a in actions:
            a.update_objective(self.total_X,self.total_Y,self.domain,target_class_idx=self.target_class_idx,transformer=self.transformer)
        return actions


    def update(self):
        self.count+=1
        self.solution_history.append(deepcopy(self.current_solution))

    def output_the_best(self,lambda_parameter=0.001):
        best_solution = max( self.solution_history,key=lambda x: simple_objective(x,self.total_X,self.total_Y,lambda_parameter=lambda_parameter,target_class_idx=self.target_class_idx) )

        return best_solution

    def output(self):
        return self.output_the_best(lambda_parameter=self.lambda_parameter)

    def compute_accuracy(self,rule_set):
        theta = len(get_correct_cover_ruleset(rule_set,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)) * 1.0 / len(self.total_X)
        return theta
