# import some critical packages we use.
import pandas as pd
import numpy as np
from fim import fpgrowth
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csc_matrix,csr_matrix

# import standard programming-purpose package
import time
from copy import deepcopy
from itertools import chain, combinations
import itertools
from random import sample
import operator
from collections import Counter, defaultdict
from numpy.random import random
from bisect import bisect_left

# import math
from math import lgamma
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom

# import local package
from .util import table_to_binary_df

class BRS(object):
    '''
    The Bayesian Decision Set model
    '''
    def __init__(self,data_table, blackbox,target_class_idx=1):
        self.data_table = data_table
        self.blackbox = blackbox
        self.target_class_idx = target_class_idx

        self.discretized_data_table,self.df,self.Y = table_to_binary_df(data_table)
        self.attributeLevelNum = defaultdict(int)
        self.attributeNames = []
        for i,attribute in enumerate( self.discretized_data_table.domain.attributes):
            self.attributeNames.append(attribute.name)
            self.attributeLevelNum[attribute.name] = len(attribute.values)
        self.N = len(self.Y)

    def fit(self,domain,X,Y=None,Niteration=100):
        """
        note that if the BRS keeps producing small size rule set,
        set the the max_len (maxmum length of a rule) to be larger
        """
        N = 2000      # number of rules to be used in SA_patternbased and also the output of generate_rules
        Niteration = 500  # number of iterations in each chain
        Nchain = 2         # number of chains in the simulated annealing search algorithm

        # supp = 0.05           # 5% is a generally good number. The higher this supp, the 'larger' a pattern is
        supp = 0.01
        # supp = 0.008
        # maxlen = 5       # maxmum length of a pattern

        #
        maxlen = min(7,X.shape[1])

        # \rho = alpha/(alpha+beta). Make sure \rho is close to one when choosing alpha and beta.
        # alpha_1 = 500       # alpha_+
        # beta_1 = 1          # beta_+
        # alpha_2 = 500         # alpha_-
        # beta_2 = 1       # beta_-

        # see this in the hyper-parameter tuning section of the BRS paper
        alpha_1 = 900
        beta_1 = 100
        alpha_2 = 900
        beta_2 = 100

        min_support = 10
        p_data = np.where(self.Y==self.target_class_idx)[0]
        print(p_data.shape)
        supp_num = max( int(p_data.shape[0]*supp), min_support)
        # supp_num = 100


        print("start BRS algorithm")

        self.generate_rules(supp_num,maxlen,N,method='fpgrowth')
        self.set_parameters(alpha_1,beta_1,alpha_2,beta_2,None,None)
        rules = self.SA_patternbased(Niteration,Nchain,print_message=False)
        self.rules = rules
        return rules

    def getPatternSpace(self):
        print ('Computing sizes for pattern space ...')
        start_time = time.time()
        """ compute the rule space from the levels in each attribute """
        for item in self.attributeNames:
            self.attributeLevelNum[item+'_neg'] = self.attributeLevelNum[item]
        self.patternSpace = np.zeros(self.maxlen+1)
        tmp = [ item + '_neg' for item in self.attributeNames]
        self.attributeNames.extend(tmp)
        for k in range(1,self.maxlen+1,1):
            for subset in combinations(self.attributeNames,k):
                tmp = 1
                for i in subset:
                    tmp = tmp * self.attributeLevelNum[i]
                self.patternSpace[k] = self.patternSpace[k] + tmp
        print ('\tTook %0.3fs to compute patternspace' % (time.time() - start_time))

    def generate_rules(self,supp=2,maxlen=10,N=10000,method='fpgrowth'):
        '''
        fp-growth, apriori, or using a tree based method.
        Note that, for frequen itemset mining, data needs to be discretized first.
        '''
        self.maxlen = maxlen
        self.supp = supp
        df = 1-self.df #df has negative associations
        df.columns = [name.strip() + '_neg' for name in self.df.columns]
        df = pd.concat([self.df,df],axis = 1)
        # if method =='fpgrowth' and maxlen<=3:
        if method =='fpgrowth' and self.maxlen <= 3:
        # if method =='fpgrowth':
            print ('Generating rules using fpgrowth, of support',self.supp,"max len", self.maxlen)
            # itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]
            cols = df.columns.values.astype(str)
            R,C = np.where(df.values==1)
            itemMatrix = np.split(cols[C],np.unique(R,return_index=True)[1])[1:]
            itemMatrix = [ item.tolist() for item in itemMatrix]


            pindex = np.where(self.Y==self.target_class_idx)[0]
            nindex = np.where(self.Y!=self.target_class_idx)[0]
            # pindex = np.where(self.Y==1)[0]
            # nindex = np.where(self.Y!=1)[0]
            start_time = time.time()
            rules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = self.maxlen)
            rules = [tuple(np.sort(rule[0])) for rule in rules]
            rules = list(set(rules))
            print ('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)))
        else:
            rules = []
            print ('Generating rules using tree-based method ')
            start_time = time.time()
            for length in range(1,maxlen+1,1):
                n_estimators = min(pow(df.shape[1],length),300)
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],df.columns,target_class_idx=self.target_class_idx))
            rules = [list(x) for x in set(tuple(x) for x in rules)]
            print ('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules)) )
        self.screen_rules(rules,df,N) # select the top N rules using secondary criteria, information gain
        self.getPatternSpace()

    def screen_rules(self,rules,df,N):
        print ('Screening rules using information gain')
        start_time = time.time()
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = i
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        len_rules = [len(rule) for rule in rules]
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        print("1")
        df_array = df.to_numpy()
        print("2")
        # matrix = np.matrix(df_array)
        matrix = csr_matrix(df_array)
        print("2.5")
        print(matrix.shape)
        print(ruleMatrix.shape)
        mat = matrix * ruleMatrix
        # mat = np.matrix(df) * ruleMatrix
        print("3")
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)
        print("4")
        del mat,lenMatrix,indptr,data,len_rules,indices,itemInd
        Zpos = [Z[i] for i in np.where(self.Y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        # supp_select = np.where(TP>=self.supp*sum(self.Y)/100)[0]
        # supp_select = np.where(TP>=0)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        supp_select = np.where(TP>=FP)[0]
        TN = len(self.Y) - np.sum(self.Y) - FP
        FN = np.sum(self.Y) - TP
        p1 = TP.astype(float)/(TP+FP)
        p2 = FN.astype(float)/(FN+TN)
        pp = (TP+FP).astype(float)/(TP+FP+TN+FN)
        tpr = TP.astype(float)/(TP+FN)
        fpr = FP.astype(float)/(FP+TN)
        cond_entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
        cond_entropy[p1*(1-p1)==0] = -((1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2)))[p1*(1-p1)==0]
        cond_entropy[p2*(1-p2)==0] = -(pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1)))[p2*(1-p2)==0]
        cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
        select = np.argsort(cond_entropy[supp_select])[::-1][-N:]
        print("5")
        print(select.shape)
        self.rules = [rules[i] for i in supp_select[select]]
        self.RMatrix = np.array(Z[:,supp_select[select]])
        print ('\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(self.rules)) )

    def set_parameters(self, a1=100,b1=1,a2=1,b2=100,al=None,bl=None):
        # input al and bl are lists
        self.alpha_1 = a1
        self.beta_1 = b1
        self.alpha_2 = a2
        self.beta_2 = b2
        if al ==None or bl==None or len(al)!=self.maxlen or len(bl)!=self.maxlen:
            print ('No or wrong input for alpha_l and beta_l. The model will use default parameters!')
            self.C = [1.0/self.maxlen for i in range(self.maxlen)]
            self.C.insert(0,-1)
            self.alpha_l = [10 for i in range(self.maxlen+1)]
            self.beta_l= [10*self.patternSpace[i]/self.C[i] for i in range(self.maxlen+1)]
        else:
            self.alpha_l=al
            self.beta_l = bl

    def SA_patternbased(self, Niteration = 5000, Nchain = 3, q = 0.1, init = [], print_message=True):
        # print 'Searching for an optimal solution...'
        start_time = time.time()
        nRules = len(self.rules)
        self.rules_len = [len(rule) for rule in self.rules]
        maps = defaultdict(list)
        T0 = 1000
        split = 0.7*Niteration
        for chain in range(Nchain):
            # initialize with a random pattern set
            if init !=[]:
                rules_curr = init[:]
            else:
                N = sample(range(1,min(8,nRules),1),1)[0]
                rules_curr = sample(range(nRules),N)
            rules_curr_norm = self.normalize(rules_curr)
            pt_curr = -100000000000
            maps[chain].append([-1,[pt_curr/3,pt_curr/3,pt_curr/3],rules_curr,[self.rules[i] for i in rules_curr]])
            for iter in range(Niteration):
                if iter>=split:
                    p = np.array(range(1+len(maps[chain])))
                    p = np.array(list(accumulate(p)))
                    p = p/p[-1]
                    index = find_lt(p,random())
                    rules_curr = maps[chain][index][2][:]
                    rules_curr_norm = maps[chain][index][2][:]
                rules_new, rules_norm = self.propose(rules_curr[:], rules_curr_norm[:],q)
                cfmatrix,prob =  self.compute_prob(rules_new)
                T = T0**(1 - iter/Niteration)
                pt_new = sum(prob)
                alpha = np.exp(float(pt_new -pt_curr)/T)

                if pt_new > sum(maps[chain][-1][1]):
                    maps[chain].append([iter,prob,rules_new,[self.rules[i] for i in rules_new]])
                    if print_message:
                        print ('\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(chain, iter,(cfmatrix[0]+cfmatrix[2]+0.0)/len(self.Y),cfmatrix[0],cfmatrix[1],cfmatrix[2],cfmatrix[3],sum(prob), prob[0], prob[1], prob[2]) )
                        # print '\n** chain = {}, max at iter = {} ** \n obj = {}, prior = {}, llh = {} '.format(chain, iter,prior+llh,prior,llh)
                        self.print_rules(rules_new)
                        print (rules_new)
                if random() <= alpha:
                    rules_curr_norm,rules_curr,pt_curr = rules_norm[:],rules_new[:],pt_new
        pt_max = [sum(maps[chain][-1][1]) for chain in range(Nchain)]
        index = pt_max.index(max(pt_max))
        # print '\tTook %0.3fs to generate an optimal rule set' % (time.time() - start_time)
        return maps[index][-1][3]

    def propose(self, rules_curr,rules_norm,q):
        nRules = len(self.rules)
        Yhat = (np.sum(self.RMatrix[:,rules_curr],axis = 1)>0).astype(int)
        incorr = np.where(self.Y!=Yhat)[0]
        N = len(rules_curr)
        if len(incorr)==0:
            clean = True
            move = ['clean']
            # it means the HBOA correctly classified all points but there could be redundant patterns, so cleaning is needed
        else:
            clean = False
            incorr = incorr.tolist()
            ex = sample(incorr,1)[0]
            t = random()
            if self.Y[ex]==1 or N==1:
                if t<1.0/2 or N==1:
                    move = ['add']       # action: add
                else:
                    move = ['cut','add'] # action: replace
            else:
                if t<1.0/2:
                    move = ['cut']       # action: cut
                else:
                    move = ['cut','add'] # action: replace
        if move[0]=='cut':
            """ cut """
            if random()<q:
                candidate = list(set(np.where(self.RMatrix[ex,:]==1)[0]).intersection(rules_curr))
                if len(candidate)==0:
                    candidate = rules_curr
                cut_rule = sample(candidate,1)[0]
            else:
                p = []
                all_sum = np.sum(self.RMatrix[:,rules_curr],axis = 1)
                for index,rule in enumerate(rules_curr):
                    Yhat= ((all_sum - np.array(self.RMatrix[:,rule]))>0).astype(int)
                    TP,FP,TN,FN  = getConfusion(Yhat,self.Y)
                    p.append(TP.astype(float)/(TP+FP+1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p,0,0)
                p = np.array(list(accumulate(p)))
                if p[-1]==0:
                    index = sample(range(len(rules_curr)),1)[0]
                else:
                    p = p/p[-1]
                index = find_lt(p,random())
                cut_rule = rules_curr[index]
            rules_curr.remove(cut_rule)
            rules_norm = self.normalize(rules_curr)
            move.remove('cut')

        if len(move)>0 and move[0]=='add':
            """ add """
            if random()<q:
                add_rule = sample(range(nRules),1)[0]
            else:
                Yhat_neg_index = list(np.where(np.sum(self.RMatrix[:,rules_curr],axis = 1)<1)[0])
                mat = np.multiply(self.RMatrix[Yhat_neg_index,:].transpose(),self.Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat,axis = 1)
                FP = np.array((np.sum(self.RMatrix[Yhat_neg_index,:],axis = 0) - TP))
                TN = np.sum(self.Y[Yhat_neg_index]==0)-FP
                FN = sum(self.Y[Yhat_neg_index]) - TP
                p = (TP.astype(float)/(TP+FP+1))
                p[rules_curr]=0
                add_rule = sample(np.where(p==max(p))[0].tolist(),1)[0]
            if add_rule not in rules_curr:
                rules_curr.append(add_rule)
                rules_norm = self.normalize(rules_curr)

        if len(move)>0 and move[0]=='clean':
            remove = []
            for i,rule in enumerate(rules_norm):
                Yhat = (np.sum(self.RMatrix[:,[rule for j,rule in enumerate(rules_norm) if (j!=i and j not in remove)]],axis = 1)>0).astype(int)
                TP,FP,TN,FN = getConfusion(Yhat,self.Y)
                if TP+FP==0:
                    remove.append(i)
            for x in remove:
                rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def compute_prob(self,rules):
        Yhat = (np.sum(self.RMatrix[:,rules],axis = 1)>0).astype(int)
        TP,FP,TN,FN = getConfusion(Yhat,self.Y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength = self.maxlen+1))
        try:
            prior_ChsRules= sum([log_betabin(Kn_count[i],self.patternSpace[i],self.alpha_l[i],self.beta_l[i]) for i in range(1,len(Kn_count),1)])
        except:
            print("1")
        try:
            likelihood_1 =  log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)
        except:
            print("2")
        try:
            likelihood_2 = log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        except:
            print("3")
        return [TP,FP,TN,FN],[prior_ChsRules,likelihood_1,likelihood_2]

    def normalize_add(self, rules_new, rule_index):
        rules = rules_new[:]
        for rule in rules_new:
            if set(self.rules[rule]).issubset(self.rules[rule_index]):
                return rules_new[:]
            if set(self.rules[rule_index]).issubset(self.rules[rule]):
                rules.remove(rule)
        rules.append(rule_index)
        return rules

    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1<len(rules):
                for p2 in range(p1+1,len(rules),1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except:
            return rules_new[:]


    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print (self.rules[rule_index])

    def rules_convert(self,rules,domain):
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

            def filter_data_(self, X):
                return self.filter_data(X)

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
            def __init__(self,conditions,domain,target_class_idx):
                """
                Parameters:
                    feature_value_pairs : a list of (column_idx,value) pairs
                    class_label: the target class
                    domain: the domain
                """
                self.conditions = conditions
                self.target_class_idx = target_class_idx

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

            def evaluate_data_(self, X):
                return self.evaluate_data(X)

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

        for feature in self.data_table.domain.attributes:
            if feature.is_continuous:
                feature.max = max([ d[feature] for d in self.data_table ]).value
                feature.min = min([ d[feature] for d in self.data_table ]).value
        rule_set = []
        col_names = [it.name for it in domain.attributes]
        possible_range = [ (it.min,it.max) if it.is_continuous else (0,0) for it in self.data_table.domain.attributes   ]
        print("number of rules",len(rules))
        for rule_str_tuple in rules:
            # try:
            previous_cols =[]
            conditions=[]
            for condition in rule_str_tuple:
                name,string_conditions = condition.split('_')[:2]
                negative_flag = ( len(condition.split('_')) == 3 )
                col_idx = col_names.index(name)
                if col_idx not in previous_cols:
                    if domain.attributes[col_idx].is_continuous:
                        print("error, BRS do not handle continuous feature. Something went wrong. Please discretize first")
                        c = re.split( "- | ",string_conditions)
                        if len(c) == 2:
                            if c[0] in ['<=','≤','<']:
                                s = Condition(column=col_idx,max_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                            else:
                                s = Condition(column=col_idx,min_value=float(c[1]),type='continuous',possible_range=possible_range[col_idx])
                            conditions.append(s)
                        if len(c) == 3:
                            s = Condition(column=col_idx,min_value=float(c[0]),max_value=float(c[2]),type='continuous')
                            conditions.append(s)
                    else:
                        # categorical
                        all_values = self.data_table.domain.attributes[col_idx].values
                        string_conditions_idx = all_values.index(string_conditions)
                        if negative_flag:
                            this_values = [ i for i in range( len(all_values) ) if i!=string_conditions_idx ]
                        else:
                            this_values = [string_conditions_idx]
                        s = Condition(column=col_idx,values=this_values,type='categorical')
                        conditions.append(s)
                    previous_cols.append(col_idx)
                else:
                    s = conditions[ previous_cols.index(col_idx) ]
                    if domain.attributes[col_idx].is_continuous:
                        print("error, BRS do not handle continuous feature. Something went wrong. Please discretize first")
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
                        all_values = self.data_table.domain.attributes[col_idx].values
                        string_conditions_idx = all_values.index(string_conditions)
                        if negative_flag:
                            new_values = set( [ i for i in range( len(all_values) ) if i!=string_conditions_idx ] )
                            old_values = set(s.values[:])
                            s.values = sorted( old_values & new_values  )
                        else:
                            new_values = set ( [string_conditions_idx] )
                            # s.values.append(string_conditions_idx)
                            # old_values = set(s.values[:])
                            # s.values = sorted( old_values | new_values  )
                            s.values = sorted( new_values)

            # merge, for example, merge "1<x<2" and "2<x<3" into "1<x<3"
            rule = Rule(conditions=conditions,domain=domain,target_class_idx=self.target_class_idx)
            rule_set.append(rule)
            # except:
            #     print("one error in converting")
            #     continue

        return rule_set


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    print ('in find_lt,{}'.format(a))
    raise ValueError


def log_gampoiss(k,alpha,beta):
    import math
    k = int(k)
    return math.lgamma(k+alpha)+alpha*np.log(beta)-math.lgamma(alpha)-math.lgamma(k+1)-(alpha+k)*np.log(1+beta)


def log_betabin(k,n,alpha,beta):
    import math
    try:
        Const =  math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print ('alpha = {}, beta = {}'.format(alpha,beta))
    if isinstance(k,list) or isinstance(k,np.ndarray):
        if len(k)!=len(n):
            print ('length of k is %d and length of n is %d'%(len(k),len(n)))
            raise ValueError
        lbeta = []
        for ki,ni in zip(k,n):
            # lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
            lbeta.append(math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const
        # return math.lgamma(n+1)- math.lgamma(k+1) - math.lgamma(n-k+1) + math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN


def predict(rules,df):
    Z = [[] for rule in rules]
    dfn = 1-df #df has negative associations
    dfn.columns = [name.strip() + '_neg' for name in df.columns]
    df = pd.concat([df,dfn],axis = 1)
    for i,rule in enumerate(rules):
        Z[i] = (np.sum(df[list(rule)],axis=1)==len(rule)).astype(int)
    Yhat = (np.sum(Z,axis=0)>0).astype(int)
    return Yhat

def extract_rules(tree, feature_names,target_class_idx=1):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    labels = tree.tree_.value
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    rules = []
    for child in idx:
        this_label = np.argmax( labels[child] )
        if this_label != target_class_idx:
            continue
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules
