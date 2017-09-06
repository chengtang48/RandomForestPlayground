##################--------------
import urllib
from six.moves import cPickle as pickle
import random 
from csv import reader
from math import sqrt
from math import floor
import os
import numpy as np
import scipy.sparse as sps
from sklearn import random_projection
from tree_utilities import *
#from pandas import DataFrame
#from IPython.display import display
##################################

## A class of binary spatial-trees 
        
class flex_binary_trees(object):
    """
    A recursive data structure based on binary trees
     - at each node, it contains data, left, right child (or none if leaf), just as any other binary tree
     - it also knows its height
     - additionally, it has meta information about split direction and split threshold
     - to incorporate the use of master-slave trees (see below), it also has an optional reference
      to a master tree
     
    On splitting method
     - if rpart or cpart are used, labels must be provided
    """

    def __init__(self, data, data_indices=None, 
                 proj_design={'name':'projmat','params':{'name':'breiman','sparsity':3,'target_dim':10}}, 
                 split_design={'name':'cart', 'params':{'regress':False}}, 
                 stop_design={'name':'naive'}, 
                 height=0, labels=None, master_tree=None):
        """
        data: n by d matrix, the entire dataset assigned to the tree
        data_indices: the subset of indices assigned to this node
        proj_design: A dictionary that contains name and params of a method; 
          the method returns one or more splitting directions (projection matrix)
        split_design: A dict that contains name and params of a function;
          the function s.t. given 1D projected data, it must return the splitting threshold
        stop_rule: a boolean function of data_indices and height
        height: height of current node (root has 0 height)
        """
        assert data_indices is not None, "You must pass data indices to root!"
        self.data = data
        self.data_ind = data_indices
        ## This default assignment will cause problems in recursion!
        #if data_indices is None:
        #    self.data_ind = np.ones(data.shape[0], dtype=bool)
        self.proj_design = proj_design
        self.split_design = split_design 
        self.stop_design = stop_design
        
        self.height_ = height
        self.labels = labels
        self.master_tree = master_tree
        self.leftChild_ = None
        self.rightChild_ = None
            
            
        ## set stop rule: a boolean function
        if np.sum(self.data_ind) == 0:
            ## stop if this cell has no data
            self.isLeaf = True
        elif self.stop_design['name'] == 'naive':
            self.isLeaf = naive_stop_rule(self.data[self.data_ind,:], height=self.height_) 
            
        elif self.stop_design['name'] == 'cell_size':
            assert 'params' in self.stop_design, "Please specify stopping parameters!"
            d0 = self.stop_design['params']['diameter']
            max_h = None
            if 'max_level' in self.stop_design['params']:
                max_h = self.stop_design['params']['max_level']
            self.isLeaf = cell_size_rule(self.data[self.data_ind,:], self.height_,
                                         max_height=max_h, target_diameter=0.5*d0)
        else:
            print("You must provide a known stopping method!")
            self.isLeaf = True
            
    
    def proj_rule_function(self):
        """
        A function such that given name of a method, returns a projection vector (splitting direction)
        Can be override (user-defined)
        returns a n_features by n_projected_dim projection matrix, A
        
        Warning: Should only be executed if self.data_ind has at least ONE nonzero element
        """
        assert np.sum(self.data_ind) > 0, "The cell is empty!!"
        name_list = ['projmat', 'cyclic', 'full']
        
        method = self.proj_design['name']
        
        assert method in name_list, 'No such rule implemented in the current tree class!'
        
        if method == 'projmat':
            
            return comp_projmat(self.data[self.data_ind,:], **self.proj_design['params'])
        
        elif method == 'cyclic':
            # cycle through features using height information
            # here A is 1D
            n_features = self.data.shape[1]
            A = np.zeros(n_features)
            A[self.height_ % n_features] = 1
            return A
        else:
            # no compression, 'full'
            return np.eye(self.data.shape[1])
        
    def split_rule_function(self, A):
        """
        Given a projection matrix
        Returns the best split direction and threshold
        Warning: Should only be executed if self.data_ind has at least ONE nonzero element
        """
        assert np.sum(self.data_ind) > 0, "The cell is empty!"
        name_list = ['cart', 'median', 'median_perturb', '2means']
        
        method = self.split_design['name']
        assert method in name_list, 'No such split rule implemented in current tree class!'
        
        if 'params' in self.split_design:
            params = self.split_design['params']
        else:
            params = dict()
        
        if method == 'cart':
            return cart_split(self.data[self.data_ind,:], A, self.labels[self.data_ind], **params)
        elif method == 'median':
            return median_split(self.data[self.data_ind,:], A, **params)
        elif method == 'median_perturb':
            
            node_height = self.height_ # height of this node relative to root of the flex tree
            return median_perturb_split(self.data[self.data_ind,:], A, node_height, **params)
        else:
            return two_means_split(self.data[self.data_ind,:], A, **params)
        
    
    def buildtree(self):
        """
        Recursively build a tree starting from current node as root
        Constructs w (projection direction) and threshold for each node
        
        To execute buildtree, self.data_ind must have at least ONE non-zero entry
        """
        if self.split_design['name'] == 'cart':
            assert self.labels is not None, "You must provide data labels to execute CART!"
        if not self.isLeaf:
            ## set projection/transformation/selection matrix
            A = self.proj_rule_function()  # A can be dense or sparse matrix
    
            ## find the best split feature and the best threshold
            split_rule = self.split_design['name']
            _, self.w_, self.thres_ = self.split_rule_function(A)
            
            ## transform data to get one or more candidate features
            if sps.issparse(self.w_):
                projected_data = sps.csr_matrix.dot(self.data[self.data_ind, :], self.w_).squeeze()
            else:
                projected_data = np.dot(self.data[self.data_ind, :], self.w_) ## project data to 1-D
            
            data_indices = []
            ## data_ind always has the same size as the number of data size
            ## data_indices has the same size as the number of data in this cell
            for i in range(len(self.data_ind)):
                if self.data_ind[i] == 1:
                    data_indices.append(i)
            assert len(data_indices) == len(projected_data)
            data_indices = np.array(data_indices)
            
            ## split data into left and right
            left_indices = projected_data < self.thres_
            right_indices = projected_data >= self.thres_
            
            ## Here, it's still possible that one of the left or right indices is empty array
            assert np.sum(left_indices)+np.sum(right_indices) == len(data_indices)
            left_ind = data_indices[left_indices]
            right_ind = data_indices[right_indices]
            ##
            n_data = self.data.shape[0]
            left = np.zeros(n_data, dtype=bool)
            if list(left_ind):
                # make assingment only if left_ind is non-empty
                left[left_ind] = 1
            right = np.zeros(n_data, dtype=bool)
            if list(right_ind):
                right[right_ind] = 1
            
            ## build subtrees on left and right partitions
            ## By our choice, empty cell will still make a node
            self.leftChild_ = flex_binary_trees(self.data, left, self.proj_design, 
                                                    self.split_design, self.stop_design, 
                                                    self.height_+1, self.labels)
            self.leftChild_.buildtree()
            
            self.rightChild_ = flex_binary_trees(self.data, right, self.proj_design, 
                                                     self.split_design, self.stop_design, 
                                                     self.height_+1, self.labels)
            self.rightChild_.buildtree()
            
        
    def train(self):
        self.buildtree()
        
    def predict_one(self, point, predict_type='class'):
        return predict_one_bt(self, point, predict_type=predict_type)
        
    def predict(self, test, predict_type='class'):
        return predict_bt(self, test, predict_type=predict_type)
            
            
####-------- Utility functions for binary trees   

def retrievalLeaf(btree, query):
    """
    Given a binary partition tree
    returns a node that contains query
    Here, the cell of a returned leaf node may be empty
    """
    if btree.leftChild_ is None and btree.rightChild_ is None:
        return btree
    if sps.issparse(btree.w_):
        val = sps.csr_matrix.dot(btree.w_.T, query).squeeze()
    else:
        val = np.dot(btree.w_, query)
    if val < btree.thres_:
        return retrievalLeaf(btree.leftChild_, query)
    else:
        return retrievalLeaf(btree.rightChild_, query)
    
def retrievalSet(btree, query):
    """
    Given a binary partition tree
    returns indices of points in the cell that contains query point
    By design, this set of indices will NOT be all zeros
    """
    ## base case: return data indices if leaf node is reached    
    if btree.leftChild_ is None and btree.rightChild_ is None:
        ## By our recursion, data_ind returned won't be all zeros
        assert np.sum(btree.data_ind) != 0, "Something is wrong!"
        return btree.data_ind
        
    ## check which subset does the query belong to
    if sps.issparse(btree.w_):
        val = sps.csr_matrix.dot(btree.w_.T, query)
    else:
        val = np.dot(btree.w_, query)
        
    if val < btree.thres_: 
        if (btree.leftChild_ is None) or (np.sum(btree.leftChild_.data_ind)==0):
            # use parent cell as the retrieval set 
            # parent cell is guaranteed to be non-empty
            return btree.data_ind
        return retrievalSet(btree.leftChild_, query)
    else:
        if (btree.rightChild_ is None) or (np.sum(btree.rightChild_.data_ind)==0):
            return btree.data_ind
        return retrievalSet(btree.rightChild_, query)
        
def getDepth(btree, depth):
    """
    find out the depth (maximal height of all branch) of a binary tree
    depth is the depth of current node
    """
    ## via DFS
    if btree.leftChild_ is None and btree.rightChild_ is None:
        return depth
        
    depthL = getDepth(btree.leftChild_, depth+1)
    depthR = getDepth(btree.rightChild_, depth+1)
        
    if depthL >= depthR:
        return depthL
    else:
        return depthR
        
def predict_one_bt(btree, point, predict_type='class'):
    assert list(btree.labels), "This tree has no associated data labels!"
    set_ind = retrievalSet(btree, point) 
    
    if predict_type == 'class':
        if np.sum(set_ind)==0:
            # using retrievalSet makes sure set_ind is not all zeros
            # but this check is just in case
            return round(np.mean(btree.labels))
        return round(np.mean(btree.labels[set_ind]))
        
    else:
        # regression
        if np.sum(set_ind)==0:
            return np.mean(btree.labels)
        return np.mean(btree.labels[set_ind])
        
    
def predict_bt(btree, test, predict_type='class'):
    """
    Returns a list of predictions corresponding to test set
    """
    predictions = list()
    for point in test:
        predictions.append(predict_one_bt(btree, point, predict_type=predict_type))
    return predictions

        
def k_nearest(tree, query):
    """
    Given dataset organized in a tree structure  ----- TO BE IMPLEMENTED
    find the approximate k nearest neighbors of query point
    """
    return k_nearest
        
    
###
def printPartition(tree, level):
    """
    Starting from root of the tree, traverse each node at given level
    and print the partitioning it holds
    Can be used for testing purposes
    """
    if tree.height_ == level or (tree.leftChild_ is None and tree.rightChild_ is None):
        ind_set = []
        for i in range(len(tree.data_ind)):
            if tree.data_ind[i] == 1:
                ind_set.append(i)
        print(ind_set)
    else:
        printPartition(tree.leftChild_, level)
        printPartition(tree.rightChild_, level)
        
def traverseLeaves(tree):
    """
    This traversal works for both balanced and unbalanced binary trees
    """
    if tree.leftChild_ is None and tree.rightChild_ is None:
        # leaf node
        yield tree
    
    if tree.leftChild_ is not None:
        for t in traverseLeaves(tree.leftChild_):
            yield t
    if tree.rightChild_ is not None:
        for t in traverseLeaves(tree.rightChild_):
            yield t
            
#######------- A class of master trees inspired by Kpotufe's adaptive tree structure
## these are not binary trees but are "meta-trees" built on binary trees
## it is used to prune a binary tree as on-the-fly

class master_trees(object):
    
    def __init__(self, data, labels=None, parent_slave_tree=None, child_slave_tree_params=None, 
                 curr_height=0, max_height=10):
        """
        Each instance of this class has two links: a link to a parent slave tree and a link to
        a child slave tree; each slave tree is an instance of the flex_binary_tree class
        Exceptions:
            -Leaf nodes only have parent slave trees
            -Root node only has a child slave tree
        
        data: data of the entire dataset
        (unlike flex_binary_tree, each node of the master tree doesn't have data_indices as attribute; it is because
        this information is already contained in its slave tree;
        in fact, there is no need to use "data" as attribute either?)
        parent_slave_tree: slave_tree that leads to the creation of this master_tree 
            - if None, master tree is root 
            - the parent slave tree also has a pointer to this master tree
        child_slave_tree_params: a dict of parameters of the slave tree if default (RP-tree) is not used
            - if not None, it should has keys "proj_design", "split_design", "stop_design"
        child_slave_tree: slave tree that is constructed by this master tree, by calling grow_child
            - the child slave tree also has a pointer to this master tree(is this necessary?)
        curr_height: absolute level of this master tree node, counting internal levels in slave trees
        max_height: maximal level that is allowed to be reached by the entire tree 
            - used independent of the slave tree stopping test
        """
        self.data = data
        self.labels = labels
        self.slave_tree_params=child_slave_tree_params
        self.n_rep = 0
        if self.slave_tree_params is not None and 'repeat' in self.slave_tree_params:
            self.n_rep = self.slave_tree_params['repeat']
        self.curr_height = curr_height
        self.max_height = max_height
        self.leaves_list = list()
        self.parent_slave_tree = None
        
        ## initialize parent slave tree if exists
        if parent_slave_tree is not None:
            self.parent_slave_tree = parent_slave_tree
            self.parent_slave_tree.master_tree = self
        ## for testing purpose
        print_diam_leaves(self)
        
    def grow_child(self):
        """
        Method to grow a child slave tree from root to leaves
        """ 
        if self.parent_slave_tree is None and self.slave_tree_params is None:
            # default parameters
            data_ind = np.ones(self.data.shape[0], dtype=bool)
            proj_design = {'name':'projmat', 'params':{'name':'dasgupta'}}
                
            ## Here, data diameter is the diameter of the entire dataset (since this is the root of master tree)
            ddiameter,_,_ = data_diameter(self.data) 
            #print(self.data.shape)
            split_design = {'name':'median_perturb', 'params':{'diameter':ddiameter}}
            stop_design = {'name':'cell_size'}
            stop_design['params']={'diameter':ddiameter}
            #print("pass")
            
        elif self.parent_slave_tree is None and self.slave_tree_params is not None:
            # user-defined child_slave_tree params
            data_ind = np.ones(self.data.shape[0], dtype=bool)
            proj_design = self.slave_tree_params['proj_design']
            split_design = self.slave_tree_params['split_design']
            stop_design = self.slave_tree_params['stop_design']
        elif self.parent_slave_tree is not None:
            # this will be invoked as long as this master tree is not a root
            # if a parent slave tree exists, 
            # we always use its params to define the new child slave tree
            data_ind = self.parent_slave_tree.data_ind
            proj_design = self.parent_slave_tree.proj_design
            stop_design = self.parent_slave_tree.stop_design
            #print("Current diameter stored is %f" %stop_design['params']['diameter'])
            split_design = self.parent_slave_tree.split_design
            stop_design = self.parent_slave_tree.stop_design
        else:
            print("Something is wrong")
        
        ## create child slave tree
        # By design, we use the relative height in child slave tree
        self.child_slave_tree = flex_binary_trees(self.data, data_indices=data_ind, proj_design=proj_design,
                                        split_design=split_design, stop_design=stop_design,
                                        height=0, labels=self.labels, master_tree=self)
        ### employs child slave tree      
        self.child_slave_tree.buildtree()
        depth = getDepth(self.child_slave_tree, 0) #get relative depth of the child slave tree
        print("The relative depth of the new child tree is %d" %depth)
        for i in range(self.n_rep):
            ## select tree with shortest depth
            slave_tree = flex_binary_trees(self.data, data_ind, proj_design=proj_design, split_design=split_design, 
                              stop_design=stop_design, height=0, labels=self.labels, master_tree=self)
            slave_tree.buildtree()
            if depth > getDepth(slave_tree, 0):
                self.child_slave_tree = slave_tree
        
        
    def iter_child_leaves(self):
        return traverseLeaves(self.child_slave_tree)
        
    def build_master_trees(self):
        self.grow_child()
        leaves_gen = self.iter_child_leaves() # up to this point, nodes are binary trees
        
        for leaf in leaves_gen:
            ## by design, curr_height is the absolute height of this master node (counting slave tree levels)
            # Update the diameter of data in leaf
            ddiam,_,_ = data_diameter(self.data[leaf.data_ind])
            if 'params' in leaf.split_design and 'diameter' in leaf.split_design['params']:
                leaf.split_design['params']['diameter'] = ddiam
            if 'params' in leaf.stop_design and 'diameter' in leaf.stop_design['params']:
                leaf.stop_design['params']['diameter'] = ddiam
            
            # child tree leaves will become parent slave trees for the new master tree
            new_master_tree = master_trees(self.data, labels=self.labels, parent_slave_tree=leaf, 
                                            curr_height=leaf.height_+self.curr_height ,max_height=self.max_height )
            #display(new_master_tree.parent_slave_tree)
            self.leaves_list.append(new_master_tree)
            
        ## Decide whether to grow the next level of master trees
        if not mtree_test_stop(self.leaves_list, self.max_height):
            for mleaf in self.leaves_list:
                if np.sum(mleaf.parent_slave_tree.data_ind) > 1:
                    ## if max height is not achieved in any cell of slave tree
                    # and that the cell is not empty, continue grow it
                    print("A new master tree built!")
                    mleaf.build_master_trees()
                
    def train(self):
        ## interface with cross-validation evalutaion
        self.build_master_trees()
        
    def predict_one(self, point, predict_type='class'):
        return predict_one_mt(self, point, predict_type=predict_type)
     
    def predict(self, test, predict_type='class'):
        ## interface with cross-validation evaluation
        return predict_mt(self, test, predict_type=predict_type)

            

####----------- Utility functions for master trees
def mtree_test_stop(mtree_leaves_list, max_height):
    count=1
    for mtree_leaf in mtree_leaves_list:
        if mtree_leaf.curr_height >= max_height:
            #print("Stop test is passed at height %d" %mtree_leaf.curr_height)
            return True
        print("Depth of the %d-th leaf is %d" %(count,mtree_leaf.curr_height))
        data_ind = mtree_leaf.parent_slave_tree.data_ind
        ddiam,_,_ = data_diameter(mtree_leaf.data[data_ind,:])
        #print("Diameter of the %d-th leaf is %f" %(count, ddiam))
        count+=1
    
    return False
       
def retrievalLeaf_mtree(mtree, query):
    ## base case
    if not mtree.leaves_list:
        return mtree
    
    ##
    #find the leaf containing query using its binary slave tree
    slave_leaf = retrievalLeaf(mtree.child_slave_tree, query) 
    return retrievalLeaf_mtree(slave_leaf.master_tree, query) #recurse on master tree of the found leaf

def traverseLeaves_mtree(mtree):
    if not mtree.leaves_list:
        yield mtree
    
    for leaf in mtree.iter_child_leaves():
        traverseLeaves_mtree(leaf.master_tree)


def print_mtree_leaves(mtree):
    count = 0
    if not mtree.leaves_list:
        if mtree.parent_slave_tree is not None:
            indices = np.arange(mtree.data.shape[0])
            display(indices[mtree.parent_slave_tree.data_ind])
            #return np.sum(mtree.parent_slave_tree.data_ind)
            return count+1
        else:
            print("Root node")
            
    else:
        for leaf in mtree.leaves_list:
            count += print_mtree_leaves(leaf)
        return count

def apply_mtree_leaves(mtree, myfunc):
    if not mtree.leaves_list:
        if mtree.parent_slave_tree is not None:
            myfunc(mtree.parent_slave_tree)
        else:
            print("Root node")
    else:
        for leaf in mtree.leaves_list:
            apply_mtree_leaves(leaf)

def print_diam_leaves(mtree):
    def myfunc(f_tree):
        ddiam,_,_ = data_diameter(f_tree.data[f_tree.data_ind])
        print(ddiam)
    apply_mtree_leaves(mtree, myfunc)
    
    
def predict_one_mt(mtree, point, predict_type='class'):
    """
    In the future, change retrieval method to retrievalSet_mtree, which should have 
    better performance
    """
    mLeaf = retrievalLeaf_mtree(mtree, point)
    set_ind = mLeaf.parent_slave_tree.data_ind
    
    if predict_type == 'class':
        #print(round(np.mean(mcell.slave_tree.labels[set_ind])))
        if np.sum(set_ind)==0:
            # if set index is an empty list
            return round(np.mean(mLeaf.parent_slave_tree.labels))
        return round(np.mean(mLeaf.parent_slave_tree.labels[set_ind]))
        
    else:
        # regression
        if np.sum(set_ind)==0:
            return np.mean(mLeaf.parent_slave_tree.labels)
        return np.mean(mLeaf.parent_slave_tree.labels[set_ind])
    
def predict_mt(mtree, test, predict_type='class'):
    """
    Returns a list of predictions corresponding to test set
    """
    predictions = list()
    for point in test:
        predictions.append(predict_one_mt(mtree, point, predict_type=predict_type))
    return predictions        
                    