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

from tqdm import tqdm

def explain_tabular(dataset,encoder, blackbox, target_class='yes', pre_label=True, random_seed=42,beta=1):
    '''
    Input Params:
    1. dataset: a Orange data table
    2. encoder: a scikit transformer
    3. blackbox: a blackbox predict function, such as `c.predict` where c is a scikit-classifier
    4. target_class
    ---
    Output:
    A decision set.
    '''
    np.random.seed(random_seed)

    if pre_label == False:
        # re-labelled the data using the blackbox, otherwise assuming the labels provided is labeled by the classifier, (instead of the groundtruth label)
        labels = blackbox(encoder(dataset.X))
        dataset = Orange.data.Table(dataset.domain, dataset.X, labels)

    # initialize with the true data instances
    explainer = ADS(dataset, blackbox, encoder, target_class=target_class) # fit the explainer to the data
    explainer.set_parameters(beta=beta) # set hyperparameter

    explainer.initialize() # initialize the decision set as empty

    explainer.initialize_synthetic_dataset() # initialize synthetic dataset as empty
    # while(explainer.termination_condition() == False):
    for _ in tqdm(range(explainer.N_iter_max)):
        # print(explainer.current_obj,len(explainer.current_solution))
        # the local search: main algorithm

        # actions = explainer.generate_action_space() # generating possible actions
        try:
            actions = explainer.generate_action_space() # generating possible actions
        except Exception as e:
            print(e)
            print("indicates actions for certain mode has no plausible actions at all")
            continue
        a_star,a_prime = best_and_second_best_action(actions)
        tmp_count = 0
        while(a_prime.upper() > a_star.lower() +0.001 ) :
            # print(a_prime.upper(), a_star.lower() )
            X_new = explainer.generate_synthetic_instances(a_star,a_prime)
            actions =  explainer.update_actions(actions,a_star,a_prime,X_new)
            a_star,a_prime = best_and_second_best_action(actions)
            tmp_count+=1
        if tmp_count != 0:
            print(tmp_count)

        explainer.update_best_solution(a_star) # update best solution
        explainer.update_current_solution(a_star,actions) #take the greedy or take random
        explainer.update() # count += 1
        del actions
        # print(len(explainer.synthetic_data_table))



    print("the result accuracy",explainer.compute_accuracy())
    rule_set = explainer.output()
    print("the number of rules",len(rule_set))
    print("the number of new instances generated",len(explainer.synthetic_data_table))
    print("okay!")
    return rule_set,explainer
