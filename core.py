# import general package function
import numpy as np
import heapq
import math
from copy import deepcopy
# import local function
# from objective import objective


def simple_objective(solution,X,Y,parameter=0.01):
    theta = len(get_correct_cover_ruleset(solution,X,Y)) * 1.0 / len(X)
    return theta - parameter * len(solution)

def get_incorrect_cover_ruleset(solution,X,Y):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    return np.where(Y.astype(np.bool)!=curr_covered_or_not)[0]

def get_correct_cover_ruleset(solution,X,Y):
    curr_covered_or_not = np.zeros(X.shape[0], dtype=np.bool)
    for r in solution:
        curr_covered_or_not |= r.evaluate_data(X)
    return np.where(Y.astype(np.bool)==curr_covered_or_not)[0]

def best_and_second_best_action(actions):
    try:
        tmp = heapq.nlargest(2,actions)
        return tmp[0],tmp[1]
    except:
        print("strang thing happened, there is only one action")
        null_action = deepcopy(tmp[0]); null_action.make_hopeless()
        return tmp[0],


# Helper function for smooth_local_search routine: Computes the 'estimate' of optimal value using random search
def compute_OPT(list_rules, X, Y, lambda_array):
    opt_set = set()
    for i in range(len(list_rules)):
        r_val = np.random.uniform()
        if r_val <= 0.5:
            opt_set.add(i)
    return objective(opt_set, list_rules, X, Y, lambda_array)

# Helper function for smooth_local_search routine: Computes estimated gain of adding an element to the solution set
def estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, error_threshold):
    #assumes rule_x_index is not in soln_set

    Exp1_func_vals = []

    Exp2_func_vals = []

    while(True):

        # first expectation term (include x)
        for i in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            temp_soln_set.add(rule_x_index)
            Exp1_func_vals.append(objective(temp_soln_set, list_rules, df, Y, lambda_array))

        # second expectation term (exclude x)
        for j in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            if rule_x_index in temp_soln_set:
                temp_soln_set.remove(rule_x_index)
            Exp2_func_vals.append(objective(temp_soln_set, list_rules, df, Y, lambda_array))

        # compute standard error of mean difference
        variance_Exp1 = np.var(Exp1_func_vals, dtype=np.float64)
        variance_Exp2 = np.var(Exp2_func_vals, dtype=np.float64)
        std_err = math.sqrt(variance_Exp1/len(Exp1_func_vals) + variance_Exp2/len(Exp2_func_vals))
        print("Standard Error "+str(std_err))

        if std_err <= error_threshold:
            break

    return np.mean(Exp1_func_vals) - np.mean(Exp2_func_vals)


# Helper function for smooth_local_search routine: Samples a set of elements based on delta
def sample_random_set(soln_set, delta, len_list_rules):
    all_rule_indexes = set(range(len_list_rules))
    return_set = set()

    # sample in-set elements with prob. (delta + 1)/2
    p = (delta + 1.0)/2
    for item in soln_set:
        random_val = np.random.uniform()
        if random_val <= p:
            return_set.add(item)

    # sample out-set elements with prob. (1 - delta)/2
    p_prime = (1.0 - delta)/2
    for item in (all_rule_indexes - soln_set):
        random_val = np.random.uniform()
        if random_val <= p_prime:
            return_set.add(item)

    #print(soln_set)
    #print(all_rule_indexes - soln_set)
    return return_set
