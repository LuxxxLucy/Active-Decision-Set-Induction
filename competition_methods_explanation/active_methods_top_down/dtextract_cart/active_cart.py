from copy import deepcopy
import numpy as np
import heapq
import math

from .cart import CART
from .utils import Rule,Condition
from .core import sampling_new_instance_uniform,sampling_new_instance_from_distribution


class Active_CART(CART):

    def __init__(self,data_table=None,domain=None,Y_class_idx=None,blackbox=None,root=None,current_depth=0,
        active_instance_number=0,use_real_instance_when_active = False,max_iteration=1000,record_synthetic = False):
        super().__init__(tree = 'cls',criterion = 'gini',max_depth = 100)
        if root == None:
            '''which means it is the root node'''
            self.root = self
            self.domain = domain
            self.data_table = data_table

            for feature in self.domain.attributes:
                if feature.is_continuous:
                    feature.max = max([ d[feature] for d in self.data_table ]).value
                    feature.min = min([ d[feature] for d in self.data_table ]).value
            self.Y_class_idx = Y_class_idx
            self.blackbox = blackbox
            self.rule = Rule(conditions=[],target_class_idx=Y_class_idx)
            self.depth = current_depth
            self.root.K = 0
            self.active_instance_number = active_instance_number
            self.use_real_instance_when_active = use_real_instance_when_active
            self.max_depth = 30
            self.max_iteration = max_iteration
            # self.max_iteration = 1000
            self.root.min_samples_leaf = 2

            self.record_synthetic = record_synthetic

        else:
            self.root = root
            self.rule = None
            self.K = self.root.K
            self.root.K+=1

        return

    def fit(self,X,Y,verbose=True):

        if self.active_instance_number >0:
            print("start estimate the distribution")
            # from .core import create_sampler
            # domain = self.domain
            # is_continuous_mask = np.array([ 1 if a.is_continuous else 0 for a in domain.attributes])
            # is_categorical_mask = np.array([ 1 if a.is_discrete else 0 for a in domain.attributes])
            # is_int_mask = np.array([ 0 for a in domain.attributes])
            # ranges = None
            # cov_factor=1.0
            # random_seed=42
            # self.sampler = create_sampler(X, is_continuous_mask, is_categorical_mask, is_int_mask, ranges, cov_factor, seed=random_seed)
            print("using gaussian distribution")
            from sklearn.mixture import GaussianMixture
            self.sampler = GaussianMixture()
            self.sampler.fit(X,Y)
            print("finish estimate the distribution")
            if self.record_synthetic == True:
                self.synthetic_X = np.zeros((0, X.shape[1] ))
                self.synthetic_Y = np.zeros((0))
        else:
            self.sampler = None

        self.real_X = X
        self.real_Y = Y

        self.queue_split = []
        self.id_to_node = {}
        heapq.heapify(self.queue_split)

        if(self.tree == 'reg'):
            self.criterion = 'mse'

        if verbose == True:
            # from tqdm import tqdm
            from tqdm import tqdm_notebook as tqdm
            pbar = tqdm(total=self.max_iteration)

        self.K_solved = 0

        self.root._grow_tree(X, Y, self.criterion)
        heapq.heappush(self.queue_split, (-self.gain,self.root.K)  ) # note that the gain is minus
        self.id_to_node[self.root.K] = self.root
        while len(self.queue_split)>0 and self.K_solved <= self.max_iteration:
            _, node_id = heapq.heappop(self.queue_split)
            self.K_solved+=1
            node_to_split = self.id_to_node[node_id]
            # print(node_id,node_to_split.gain)
            if node_to_split._split_tree(self.criterion) == True:
                assert node_to_split.left is not None and node_to_split.right is not None,"error!"
                gain,id,node = -node_to_split.left.gain,node_to_split.left.K,node_to_split.left
                self.id_to_node[id]=node
                heapq.heappush(self.queue_split, (gain,id) )
                gain,id,node = -node_to_split.right.gain,node_to_split.right.K,node_to_split.right
                self.id_to_node[id]=node
                heapq.heappush(self.queue_split, (gain,id) )

            assert (node_to_split.feature is None and node_to_split.left is None) or (node_to_split.feature is not None and  node_to_split.left is not None),"error!"
            pbar.update(1)

        if verbose==True:
            pbar.close()

        self.root._prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)

    def _grow_tree(self, X, Y, criterion = 'gini'):
        '''
        The essence part of active version CART is here.
        Here we wish to query the blackbox and get new samples.
        '''

        size = self.root.active_instance_number
        if size != 0:
            # import time
            # start_time = time.time()
            new_generated_X = sampling_new_instance_from_distribution(self.rule,self.root.sampler,self.root.domain,population_size=size)
            # new_generated_X = sampling_new_instance_uniform(self.rule,self.root.domain,population_size=size)
            # print ('\tTook %0.3fs to sample' % (time.time() - start_time ) )

            new_generated_Y = self.root.blackbox(new_generated_X)
            if self.root.use_real_instance_when_active == True:
                X = np.concatenate((X, new_generated_X))
                Y = np.concatenate((Y, new_generated_Y))
            else:
                X = new_generated_X
                Y = new_generated_Y
                if self.root.record_synthetic == True:
                    self.root.synthetic_X = np.concatenate((self.root.synthetic_X, X))
                    self.root.synthetic_Y = np.concatenate((self.root.synthetic_Y, Y))

        self.X = X
        self.Y = Y

        label,max_portion = max([( c ,len(Y[Y == c])*1.0/Y.shape[0]  ) for c in np.unique(Y)], key = lambda x : x[1])
        # if max_portion > 0.99 or self.depth >= self.root.max_depth or Active_CART.K >=30:
        if len(np.unique(Y)) == 1 or self.depth >= self.root.max_depth or np.sum( self.rule.evaluate_data(self.real_X) ) <= self.root.min_samples_leaf :
        # if len(np.unique(Y)) == 1 :
            self.feature = None
            self.label = label
            self.gain = 0
            return


        self.n_samples = X.shape[0]


        if criterion in {'gini', 'entropy'}:
            self.label = max([(c, len(Y[Y == c])) for c in np.unique(Y)], key = lambda x : x[1])[0]
        else:
            self.label = np.mean(Y)

        impurity_node = self._calc_impurity(criterion, Y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for col in range(X.shape[1]):
            feature_level = np.unique(X[:,col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                Y_l = Y[X[:,col] <= threshold]
                impurity_l = self._calc_impurity(criterion, Y_l)
                n_l = float(Y_l.shape[0]) / self.n_samples

                Y_r = Y[X[:,col] > threshold]
                impurity_r = self._calc_impurity(criterion, Y_r)
                n_r = float(Y_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        if best_gain <= 0:
            self.gain = 0
            return

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        # self._split_tree(X, Y, criterion)

    def _split_tree(self,criterion):

        if self.feature is None:
        # print(real_instance_number)
        # print(self.rule.conditions)):
            return False

        X, Y = self.X,self.Y

        X_l = X[X[:, self.feature] <= self.threshold]
        Y_l = Y[X[:, self.feature] <= self.threshold]
        self.left = Active_CART(root=self.root,current_depth=self.depth+1)
        self.left.rule = deepcopy(self.rule); self.left.rule.add_constraint(self.feature,self.threshold,domain=self.root.domain,mode="left_child")
        # self.left.depth = self.depth + 1
        self.left.real_X = self.real_X[ self.real_X[:,self.feature] <= self.threshold ]
        self.left._grow_tree(X_l, Y_l, criterion)

        X_r = X[X[:, self.feature] > self.threshold]
        Y_r = Y[X[:, self.feature] > self.threshold]
        self.right = Active_CART(root=self.root,current_depth=self.depth+1)
        self.right.rule = deepcopy(self.rule); self.right.rule.add_constraint(self.feature,self.threshold,domain=self.root.domain,mode="right_child")
        # self.right.depth = self.depth + 1
        self.right.real_X = self.real_X[ self.real_X[:,self.feature] > self.threshold ]
        self.right._grow_tree(X_r, Y_r, criterion)
        return True

    def output_all(self):
        rules = []
        def _dfs(node):
            # if node.left is not None:
            #     _dfs(node.left)
            # if node.right is not None:
            #     _dfs(node.right)
            if node.feature is not None:
                _dfs(node.left)
                _dfs(node.right)
            else:
                node.rule.set_target_class_idx(int(node.label),domain=self.domain)
                rules.append(node.rule)
        _dfs(self.root)

        return [r for r in rules if r.target_class_idx == self.Y_class_idx]
