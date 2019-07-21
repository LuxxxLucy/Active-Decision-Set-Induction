import numpy as np
def rules_convert(rule_set_IDS,data_table,target_class_idx):
    import operator
    import re
    import math
    from functools import reduce

    class Condition():
        '''
        a single condition.
        It is modified based on the condition class from Orange package. (in particualr, Orange.classification.rule)
        '''
        OPERATORS = {
            # discrete, nominal variables
            # 'in': lambda x,y: all( [operator.contains(y,x_tmp) for x_tmp in x ],
            'in': lambda x,y: operator.contains(y,x),
            # continuous variables
            '<=': operator.le,
            '>=': operator.ge,
        }
        def __init__(self, column, values=None, min_value=-math.inf,max_value=math.inf,type='categorical',possible_range=[0,100]):
            '''
            differs in the number of arguments as for categorical feature and continuous features
            '''
            self.type = type
            self.column = column
            if type == 'categorical':
                # for categorical features
                self.values = values # value is a set of possible values
            elif type == 'continuous':
                self.min = max(min_value,possible_range[0]) # for continuous features
                self.max = min(max_value,possible_range[1]) # for continuous features
            else:
                print("critical error. Type of condition unspecified. Must be one of categorical or continuous ")

        def filter_instance(self, x):
            """
            Filter a single instance. Returns true or false
            """
            if self.type == 'categorical':
                return Condition.OPERATORS['in'](x[self.column], self.values)
            elif self.type == 'continuous':
                return Condition.OPERATORS['<='](x[self.column], self.max) and Condition.OPERATORS['>='](x[self.column], self.min)

        def filter_data(self, X):
            """
            Filter several instances concurrently. Retunrs array of bools
            """
            if self.type == 'categorical':
                # return Condition.OPERATORS['in'](X[:,self.column],self.values)
                try:
                    return np.logical_or.reduce(
                    [np.equal(X[:,self.column], v) for v in self.values ]
                    )
                except:
                    print(self.values)
                    print(X[:5,self.column])
                    print(np.logical_or.reduce(
                    [np.equal(X[:5,self.column], v) for v in self.values ]
                    ))
                    quit()

            elif self.type == 'continuous':
                return np.logical_and(
                Condition.OPERATORS['<='](X[:,self.column], self.max),
                Condition.OPERATORS['>='](X[:,self.column], self.min)
                )
    class Rule:
        def __init__(self,conditions,domain,target_class):
            """
            Parameters:
                feature_value_pairs : a list of (column_idx,value) pairs
                class_label: the target class
                domain: the domain
            """
            self.conditions = conditions
            self.target_class = target_class
            self.target_class_idx = domain.class_var.values.index(target_class)

        def evaluate_instance(self, x):
            """
            Evaluate a single instance.

            Parameters
            ----------
            x : ndarray
                Evaluate this instance.

            Returns
            -------
            return : bool
                True, if the rule covers 'x'.
            """
            return all(condition.filter_instance(x) for condition in self.conditions)

        def evaluate_data(self, X):
            """
            Evaluate several instances concurrently.

            Parameters
            ----------
            X : ndarray
                Evaluate this data.

            Returns
            -------
            return : ndarray, bool
                Array of evaluations.
            """
            curr_covered = np.ones(X.shape[0], dtype=bool)
            for condition in self.conditions:
                curr_covered &= condition.filter_data(X)
            return curr_covered

        def __len__(self):
            return len(self.conditions)

        def get_length(self):
            return len(self.conditions)

        def get_cover(self, X):
            res = self.evaluate_data(X)
            return np.where(res == True)[0]

        def get_correct_cover(self, X, Y):
            indices_instance_covered = self.get_cover(X)
            y_tmp = Y[indices_instance_covered]

            correct_cover = []
            for ind in range(0,len(indices_instance_covered)):
                if y_tmp[ind] == self.target_class_idx:
                    correct_cover.append(indices_instance_covered[ind])

            return correct_cover, indices_instance_covered

        def get_incorrect_cover(self, X, Y):
            correct_cover, full_cover = self.get_correct_cover(X, Y)
            return (sorted(list(set(full_cover) - set(correct_cover))))

        def __eq__(self, other):
            return self.conditions == other.conditions

    for feature in data_table.domain.attributes:
        if feature.is_continuous:
            feature.max = max([ d[feature] for d in data_table ]).value
            feature.min = min([ d[feature] for d in data_table ]).value

    import re
    target_class = data_table.domain.class_var.values[ int(target_class_idx) ]
    rule_set = []
    col_names = [it.name for it in data_table.domain.attributes]
    possible_range = [ (it.min,it.max) if it.is_continuous else (0,0) for it in data_table.domain.attributes   ]
    for rule in rule_set_IDS:
        previous_cols =[]
        conditions=[]
        this_rule_target_class = data_table.domain.class_var.values[ int(rule.class_label) ]
        if this_rule_target_class != target_class:
            ''' we only want rules about the target class (positive class)'''
            continue
        for name,string_conditions in rule.itemset:
            col_idx = col_names.index(name)
            if col_idx not in previous_cols:
                if data_table.domain.attributes[col_idx].is_continuous:
                    c = re.split( "- | ",string_conditions)
                    if len(c) == 2:
                        if c[0] in ['<=','≤','<']:
                            s = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        else:
                            s = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        conditions.append(s)
                    if len(c) == 3:
                        s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous',possible_range=possible_range[col_idx])
                        conditions.append(s)
                else:
                    # categorical
                    s = Condition(column=col_idx,values=[data_table.domain.attributes[col_idx].values.index(string_conditions)],type='categorical')
                    conditions.append(s)
            else:
                s = conditions[ previous_cols.index(col_idx) ]
                if data_table.domain.attributes[col_idx].is_continuous:
                    c = re.split( "- | ",string_conditions)
                    if len(c) == 2:
                        if c[0] in ['<=','≤','<']:
                            s_tmp = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        else:
                            s_tmp = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                        s.max = max(s.max,s_tmp.max)
                        s.min = min(s.min,s_tmp.min)
                    if len(c) == 3:
                        s.max = max(s.max,float(c[2]))
                        s.min = min(s.min,float(c[0]))
                else:
                    # categorical
                    s.values.append(data_table.domain.attributes[col_idx].values.index(string_conditions))
            previous_cols.append(col_idx)

        # merge, for example, merge "1<x<2" and "2<x<3" into "1<x<3"
        rule = Rule(conditions=conditions,domain=data_table.domain,target_class=this_rule_target_class)
        rule_set.append(rule)

    return rule_set
