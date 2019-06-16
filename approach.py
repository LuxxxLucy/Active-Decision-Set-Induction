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

def explain_tabular(dataset,encoder, blackbox, target_class='yes', pre_label=True, random_seed=42):
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
    explainer.set_parameters() # set hyperparameter

    explainer.initialize() # initialize the decision set as empty

    explainer.initialize_synthetic_dataset() # initialize synthetic dataset as empty
    while(explainer.termination_condition() == False):
        # print(explainer.current_obj,len(explainer.current_solution))
        # the local search: main algorithm
        actions = explainer.generate_action_space() # generating possible actions
        a_star,a_prime = best_and_second_best_action(actions)
        while(a_prime.upper() > a_star.lower() ):
            print("now this line should never occur since now it is only passive")
            X_new = explainer.generate_synthetic_instances(a_star,a_prime)
            explainer.update(actions,X_new)
            a_star,a_prime = best_and_second_best_action(actions)

        explainer.update_best_solution(a_star) # update best solution
        explainer.update_current_solution(a_star,actions) #take the greedy or take random
        explainer.update() # count += 1


    print(explainer.compute_accuracy())
    rule_set = explainer.output()
    print("okay!")
    return rule_set
