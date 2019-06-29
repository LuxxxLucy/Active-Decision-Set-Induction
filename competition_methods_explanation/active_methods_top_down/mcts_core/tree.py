import numpy as np

from .core import Selector, Pattern, Node
from Orange.data import Table, Domain

# base batch size, number of new query to the blackbox at each roll_out_estimate phase
BASE_BATCH_SIZE = 10

from pprint import pprint as pr

def UCT(C=0.707):
    # C = sqrt(2)
    return lambda parent_node, child_node: child_node.q_value + 2 * C * np.sqrt( 2 * np.log(parent_node.count) / (child_node.count+1) )
    # return lambda parent_node, child_node: child_node.q_value + 2 * C * np.sqrt( 2 * np.log(parent_node.count) / (child_node.count+1) ) if child_node.is_finished() != True else 0

def UCB1():
    return UCT(C=0.5)

class Tree:

    def __init__(self,dataset,target_class,blackbox=None,active=True):
        '''
        1. keep memory of some key objects:domain, blackbox
        2. build the root node
        '''

        self.domain = dataset.domain # keep record of the data domain
        if blackbox == None:
            raise Exception('critical error!:blackbox must not be None')
        self.blackbox = blackbox # the sci-kit like object's `predict` function

        self.active = active # indicate active query or not

        # the initial selectors is the set of conditions that cover every data real_instances
        # that is, [min,max] for every continuous feature and [all_values] for every categorical features
        initial_selectors =  [
            Selector(col,min_value = np.amin(dataset.X[:,col]),max_value=np.amax(dataset.X[:,col]),type='continuous' )
            if feature.is_continuous
            else Selector(col, values=feature.values,type='categorical')
            for col, feature  in enumerate(dataset.domain.attributes)
            ]
        root_node = Node(self.domain,target_class, selectors=initial_selectors, parent=None, real_instances=dataset,base_selectors=initial_selectors)
        # print(root_node.pattern.target_class_idx)
        self.root_node = root_node

        self.synthetic_data_table = Table.from_domain(domain=self.domain,n_rows=0) # synthetic data (now empty)
        return

    def select_best_node(self,criteria=UCB1()):
        '''
        recursively find best child until ...
        '''
        def DFS(node):
            if node.is_terminal() and not node.is_finished():
                return node
            pairs = [ (child_node, criteria(node,child_node)) for child_node in node.children if child_node.is_finished() != True ]
            pairs = sorted(pairs, key=lambda x: x[1],reverse=True)
            for child_node,_ in pairs:
                result = DFS(child_node)
                if result != None:
                    return result
                else:
                    continue
            # return None if all child_node fails
            return None

        return DFS(self.root_node)


    def split_or_not(self,node):
        return node.split_or_not()

    def expand(self,node):
        # find the split condition and form children nodes
        # split_okay = node.split_info_gain()
        split_okay = node.split()
        if split_okay == False:
            # this node cannot be further split...
            return

        for c in node.children:
            try:
                self.roll_out_estimate(c)
            except Exception as e:
                print('error happened!')
                node.summary()
                c.summary()
                quit()
            # print(c.q_value,c.precision,c.true_positive,c.positive)

        # Note only leaf nodes track the reference to the instances, the parent node does not trach these instances anymore for memoery-efficiency. However, it would be straightforward to gather the covered instances.
        return

    def roll_out_estimate(self,node,batch_size=BASE_BATCH_SIZE):
        if self.active == False:
            # turn off active means the roll_out_estimate will not have new generated query.
            # this indicates an immediate return
            return
        if batch_size <=0:
            # enter confidence mode. Keep sampling
            pass
        else:
            # sample a fixed size of `batch_size` sample from current pattern
            new_X = node.sample(batch_size) # uniformly sample inside a region of space (specified by the pattern)
            new_y = self.blackbox(new_X) # query the blackbox to get the label
            new_synthetic_instances = Table.from_numpy(X=new_X,Y=new_y,domain=self.domain) # create new table of synthetic_instances
            # print('before update, q_value:{0},tp:{1},p:{2},count{3},precision{4}'.format(node.q_value,node.true_positive,node.positive,node.count,node.precision  ) )
            node.update_current(new_synthetic_instances) # recompute: q, true_positive, positive etc and also append new instances to synthetic_instances set
            # print('after update, q_value:{0},tp:{1},p:{2},count{3},precision{4}'.format(node.q_value,node.true_positive,node.positive,node.count,node.precision  ) )

        self.synthetic_data_table.extend(new_synthetic_instances)

        self.updating(node,new_synthetic_instances)
        return

    def updating(self,node,new_synthetic_instances):
        # print("updating!")
        current_node = node.parent

        X,Y = new_synthetic_instances.X, new_synthetic_instances.Y
        size = X.shape[0]
        covered_instances = np.ones(size,dtype=bool)
        for selector in node.pattern.selectors:
            covered_instances &= selector.filter_data(X)
        target_covered_instances = np.logical_and(  covered_instances, Y==1)
        new_true_positive = np.sum(target_covered_instances)
        new_positive = np.sum(covered_instances)

        while(  current_node != None ) :
            # update q_value, update count....
            current_node.precision = (current_node.q_value*current_node.positive + new_true_positive ) / (current_node.positive+new_positive)
            current_node.positive += new_positive
            current_node.true_positive += new_true_positive
            current_node.count += new_positive
            current_node.q_value = current_node.precision * current_node.space_coverage
            current_node = current_node.parent

        return

    def output_all(self):
        def gather_all_terminal_DFS(node,result = None):
            if result == None:
                result=[]

            if node.is_terminal():
                if node.is_significant():
                    # hopeless pattern is not considered
                    result.append(node)
                return result

            for child_node in node.children:
                result = gather_all_terminal_DFS(child_node,result)
            # return None if all child_node fails
            return result
        return gather_all_terminal_DFS(self.root_node)

    def output_finished(self):
        '''
        output all 'good' nodes
        '''
        def gather_all_DFS(node,result = None):
            if result == None:
                result=[]

            if node.is_terminal():
                if node.is_satisfied():
                    # result.append(node.pattern)

                    result.append(node)
                return result

            for child_node in node.children:
                result = gather_all_DFS(child_node,result)
            # return None if all child_node fails
            return result

        nodes = gather_all_DFS(self.root_node)
        if nodes != []:
            print("find patterns satisfying the threshold")
            return nodes
        elif nodes == []:
            # it means there is no nodes that are satisfied
            # TODO:
            print("no pattern satisfying the threshold has been found. Output current best")
            return self.output_all()
