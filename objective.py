'''
the objective function

Mainly two functions:

1. the objective function for a rule set
2. the change of value given a rule set and an action
'''



# compute the maximum length of any rule in the candidate rule set
def max_rule_length(list_rules):
    len_arr = []
    for r in list_rules:
        len_arr.append(r.get_length())
    return max(len_arr)

# compute the number of points which are covered both by r1 and r2 w.r.t. data frame X
def overlap(r1, r2, X):
    return sorted(list(set(r1.get_cover(X)).intersection(set(r2.get_cover(X)))))

# computes the objective value of a given solution set
def objective(current_set, list_rules, X, Y, lambda_array):
    # evaluate the objective function based on rules in current set
    # set is a set of rules
    # compute f1 through f7 and we assume there are 7 lambdas in lambda_array
    f = [] #stores values of f1 through f7;

    # f0 term
    f0 = len(list_rules) - len(current_set) # |S| - size(R)
    f.append(f0)

    # f1 term
    Lmax = max_rule_length(list_rules)
    sum_rule_length = 0.0
    for rule in current_set:
        sum_rule_length += rule.get_length()

    f1 = Lmax * len(list_rules) - sum_rule_length
    f.append(f1)

    # f2 term - intraclass overlap
    sum_overlap_intraclass = 0.0
    for r1_index,r1 in enumerate(current_set):
        for r2_index,r2 in enumerate(current_set):
            if r1_index >= r2_index:
                continue
            if r1.target_class_idx == r2.target_class_idx:
                sum_overlap_intraclass += len(overlap(r1, r2,X))
    f2 = X.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_intraclass
    f.append(f2)

    # f3 term - interclass overlap
    sum_overlap_interclass = 0.0
    for r1_index,r1 in enumerate(current_set):
        for r2_index,r2 in enumerate(current_set):
            if r1_index >= r2_index:
                continue
            if r1.target_class_idx != r2.target_class_idx:
                sum_overlap_interclass += len(overlap(r1, r2,X))
    f3 = X.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_interclass
    f.append(f3)

    # f4 term - coverage of all classes
    classes_covered = set() # set
    for rule in current_set:
        classes_covered.add(rule.target_class_idx)
    f4 = len(classes_covered)
    f.append(f4)

    # f5 term - accuracy
    sum_incorrect_cover = 0.0
    for rule in current_set:
        sum_incorrect_cover += len(rule.get_incorrect_cover(X,Y))
    f5 = X.shape[0] * len(list_rules) - sum_incorrect_cover
    f.append(f5)

    #f6 term - cover correctly with at least one rule
    atleast_once_correctly_covered = set()
    for rule in current_set:
        correct_cover, full_cover = rule.get_correct_cover(X,Y)
        atleast_once_correctly_covered = atleast_once_correctly_covered.union(set(correct_cover))
    f6 = len(atleast_once_correctly_covered)
    f.append(f6)

    obj_val = 0.0
    for i in range(7):
        obj_val += f[i] * lambda_array[i]

    #print(f)
    return obj_val
