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
from structure import Decision_Set_Learner,Rule,deepcopy_decision_set
from utils import rule_to_string
from core import simple_objective,get_correct_cover_ruleset,get_symmetric_difference,sample_new_instances,core_init

class ADS_Learner(Decision_Set_Learner):
    '''
    The Active Decision Set model
    '''
    def __init__(self,data_table, blackbox,target_class='yes',seed=42,use_pre_mined=False,objective='simple'):
        random.seed(seed)
        core_init(seed,data_table)
        super().__init__(data_table,target_class,seed=42)
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
        self.use_pre_mined = use_pre_mined

        self.objective_mode = objective

        self.count=0
        # categorical_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_discrete]
        # continuous_features_idx = [i for i,a in enumerate(data_table.domain.attributes) if a.is_continuous]
        #
        # self.preprocessor = make_column_transformer( ( OneHotEncoder(categories='auto',sparse=False),categorical_features_idx),
        # (StandardScaler(), continuous_features_idx),
        #                     remainder = 'passthrough'
        #                     )
        # self.preprocessor.fit(data_table.X)
        # # self.transformer = lambda x : self.preprocessor.transform(x).toarray()
        # self.transformer = lambda x : self.preprocessor.transform(x)

        self.solution_history=[]

    def initialize_synthetic_dataset(self):
        self.synthetic_data_table = Orange.data.Table.from_domain(domain=self.domain,n_rows=0) # synthetic data (now empty)
        X,Y = self.data_table.X,self.data_table.Y
        X_prime,Y_prime = self.synthetic_data_table.X,self.synthetic_data_table.Y
        self.total_X = np.append(X, X_prime, axis=0)
        self.total_Y = np.append(Y, Y_prime, axis=0)
        # self.total_X = X[:]
        # self.total_Y = Y[:]
        return

    def reset_XY(self):
        X,Y = self.data_table.X,self.data_table.Y
        self.total_X = X[:]
        self.total_Y = Y[:]
        return

    def set_parameters(self,beta,lambda_parameter):
        self.N_iter_max = 1000
        # self.N_iter_max = 4000
        self.lambda_parameter = lambda_parameter
        self.beta = beta

        self.N_batch = 10
        self.epsilon = 0.01
        self.supp=0.05
        # print("target class is:",self.target_class,". Its index is",self.target_class_idx)
        # TODO: add hyperparameter print
        if self.objective_mode == "simple":
            self.objective = lambda solution,X,Y,target_class_idx: simple_objective(solution,X,Y,target_class_idx=target_class_idx,lambda_parameter=self.lambda_parameter)
        elif self.objective_mode == "bayesian":
            from core import bayesian_objective
            self.alpha_list = [ 1 for a in self.domain.attributes]
            self.objective = lambda solution,X,Y,target_class_idx: bayesian_objective(solution,X,Y,target_class_idx=target_class_idx,alpha_list=self.alpha_list,num_attributes=len(self.domain.attributes))
        return

    def initialize(self):
        self.termination = False

        self.current_solution = []
        self.current_obj = self.objective(self.current_solution,self.data_table.X,self.data_table.Y,target_class_idx=self.target_class_idx)
        # print("initial obj: ",self.current_obj)
        self.best_obj = deepcopy(self.current_obj)
        # self.best_solution = [ Rule(conditions=[deepcopy(c) for c in r.conditions],domain=domain,target_class_idx=target_class_idx) for r in self.current_solution];
        self.best_solution = deepcopy_decision_set(self.current_solution);

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
        self.best_obj = self.objective(self.best_solution,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)

        if self.best_obj < best_action.empirical_obj:
            self.best_obj = best_action.empirical_obj
            self.best_solution = deepcopy_decision_set(best_action.new_solution);

            # todo: add obj update in the tqdm log
            # print("new best obj:",self.best_obj,"number of rules",len(self.best_solution))
        return

    def update_current_solution(self,best_action,actions):

        t = random.random()

        if t < 0.001:
            '''re-start'''
            # print("re-staring!!!!!!!!!!!!")
            self.current_solution = []
            self.current_obj = self.objective(self.current_solution,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)
            # if hasattr(self,'rule_space'):
            #     delattr(self, 'rule_space')
            return
        elif t < self.epsilon:
            # print("take random")
            a = random.choice(actions)
        else:
            a = best_action
            if a.hopeless == True:
                # todo explain here.
                print("hopeless, at iter",self.count,"mode",a.mode)
                return
        self.current_solution = a.new_solution
        # todo: better solution than this, why could this happen?
        self.current_solution = sorted(set(self.current_solution),key=lambda x:x.string_representation)
        # self.current_obj = deepcopy(a.empirical_obj)
        self.current_obj = a.obj_estimation()
        return

    def generate_action_space(self):
        actions = self.generate_action(self.total_X,self.total_Y,beta=self.beta,rho=self.rho)


        return actions

    def generate_synthetic_instances(self,a_star,a_prime):
        X_new,Y_new = sample_new_instances(a_star,a_prime,self.total_X,self.total_Y,self.domain,self.blackbox)
        return X_new,Y_new

    def update_actions(self,actions,a_star,a_prime,X_new,Y_new):
        XY_new = Orange.data.Table(self.domain, X_new, Y_new)
        self.synthetic_data_table.extend(XY_new)
        self.total_X = np.append(self.total_X, XY_new.X, axis=0)
        self.total_Y = np.append(self.total_Y, XY_new.Y, axis=0)
        start_time = time.time()
        # end_time = time.time()
        # print ('\tTook %0.3fs to generate the KDtree' % (time.time() - start_time ) )
        self.current_obj =  self.objective(self.current_solution,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)
        for a in actions:
            a.update_current_obj(self.current_obj)
            a.update_objective(self.total_X,self.total_Y,self.domain,target_class_idx=self.target_class_idx)
        return actions

    def update(self):
        self.count+=1
        self.solution_history.append(deepcopy_decision_set(self.current_solution))

    def output_the_best(self,lambda_parameter=None):
        if lambda_parameter == self.lambda_parameter:
            index,best_solution = max( enumerate(self.solution_history),key=lambda x: self.objective(x[1],self.total_X,self.total_Y,target_class_idx=self.target_class_idx) )
            print("best solution found in iteration",index)
        else:
            if self.objective_mode == "simple":
                index,best_solution = max( enumerate(self.solution_history),key=lambda x: simple_objective(x[1],self.total_X,self.total_Y,lambda_parameter=lambda_parameter,target_class_idx=self.target_class_idx) )

                print("best solution found in iteration",index)
            else:
                print("not supported")

        return best_solution

    def output(self):
        result =  self.output_the_best(lambda_parameter=self.lambda_parameter)
        # self.finish()
        return sorted(result,key=lambda x:x.string_representation)
        # return self.best_solution

    def compute_accuracy(self,rule_set):
        theta = len(get_correct_cover_ruleset(rule_set,self.total_X,self.total_Y,target_class_idx=self.target_class_idx)) * 1.0 / len(self.total_X)
        return theta
