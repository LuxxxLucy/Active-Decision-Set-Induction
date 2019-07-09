# import general package
import numpy as np
import random
from collections import namedtuple
import operator

# import particular package we utilize
import Orange

# import local package
from model import ADS
from core import best_and_second_best_action

from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm

def explain_tabular(dataset,blackbox, target_class='yes', pre_label=True, random_seed=42,beta=1,lambda_parameter=0.01):
    '''
    Input Params:
    1. dataset: a Orange data table
    2. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    3. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)

    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(dataset.X)
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    error_log_list = []

    # initialize with the true data instances
    explainer = ADS(dataset, blackbox, target_class=target_class) # fit the explainer to the data
    explainer.set_parameters(beta=beta,lambda_parameter=lambda_parameter) # set hyperparameter

    explainer.initialize() # initialize the decision set as empty

    explainer.initialize_synthetic_dataset() # initialize synthetic dataset as empty
    # the local search: main algorithm
    for iter_number in tqdm(range(explainer.N_iter_max)):
        try:
            actions = explainer.generate_action_space() # generating possible actions
        except Exception as e:
            # print(e)
            error_log_list.append((iter_number,e))
            continue
        # actions = explainer.generate_action_space() # generating possible actions
        a_star,a_prime = best_and_second_best_action(actions)


        tmp_count = 0
        while(a_prime.upper() > a_star.lower() +0.001 ) :
            # print(a_prime.upper(),a_prime.beta_interval,a_star.lower(),a_star.beta_interval)
            # print(tmp_count)
            X_new,Y_new = explainer.generate_synthetic_instances(a_star,a_prime)
            solutions =  explainer.update_actions(actions,a_star,a_prime,X_new,Y_new)
            a_star,a_prime = best_and_second_best_action(actions)
            tmp_count+=1
            if tmp_count >= 100:
                from utils import rule_to_string
                print(a_star.mode,a_star.total_volume,a_prime.total_volume)
                print(rule_to_string(a_star.changed_rule,domain=dataset.domain, target_class_idx=1))
                print(rule_to_string(a_prime.changed_rule,domain=dataset.domain, target_class_idx=1))
                break
        if tmp_count != 0:
            print("generating new instance for a batch",tmp_count)
        explainer.update_best_solution(a_star) # update best solution
        explainer.update_current_solution(a_star,actions) #take the greedy or take random
        explainer.update() # count += 1
        del actions

    print("the result accuracy",explainer.compute_accuracy())
    rule_set = explainer.output()
    print("the number of rules",len(rule_set))
    print("the number of new instances generated",len(explainer.synthetic_data_table))
    # print("Now print Error Log")
    # if len(error_log_list) == 0:
    #     print("no error happens")
    # else:
    #     for iter_num,e in error_log_list:
    #         print("at iteration:",iter_num,"happens the following error")
    #         print(e)

    return rule_set,explainer
