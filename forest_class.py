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
from generalTrees_class import *
#from tree_utilities import *
#from pandas import DataFrame
#from IPython.display import display
##################################


class forest(object):
    def __init__(self, data, labels=None, tree_design=None, predictor_type='class',
                  n_trees=10, n_samples=100, n_features=None):
        """
        tree_design: A dictionary containing 
          - tree: specifies type of the tree; flex or master
          - a proj_design dictionary, 
          - a split_design dictionary,
          - a stop_design dictionary
        
        """
        self.data = data
        self.labels = labels
        self.tree_design = tree_design
        self.predictor_type = predictor_type
        self.n_trees = n_trees
        self.trees = list() # store trees for re-use
        self.n_samples = n_samples
        self.n_features = n_features
        
        
    def reset_sample_size(self, n_samples=None, n_features=None):
        if n_samples is not None:
            self.n_samples = n_samples
        if n_features is not None:
            self.n_features = n_features
            
    def reset_predictor_type(self, method):
        self.predictor_type = method
        
    def build_forest_classic(self, isPredict=True):
        if isPredict:
            assert self.labels is not None, "Data labels missing!"
        tree_type = self.tree_design['tree']
        
        for i in range(self.n_trees):
            # sample data points with replacement
            # note numpy indexing supports repetitive/duplicate indexing
            data_ind = np.random.choice(self.data.shape[0], self.n_samples, replace=True)
            data_tree = self.data[data_ind,:] # data unique to this tree
            
            if self.n_features is not None:
                ## optionally subsample features (note: this is NOT done in the original RF)
                feature_ind = np.random.choice(self.data.shape[1], self.n_features, replace=True)
                data_tree = data_tree[:,feature_ind] # features unique to this tree
            
            if isPredict:
                labels_tree = self.labels[data_ind]
            else:
                labels_tree = None
            
            if tree_type == 'flex':
                proj_design, split_design, stop_design = self.tree_design['proj_design'],\
                   self.tree_design['split_design'],self.tree_design['stop_design']
                    
                f_tree = flex_binary_trees(data_tree, np.ones(data_tree.shape[0], dtype=bool), 
                                        proj_design,split_design, stop_design, labels=labels_tree)
                f_tree.buildtree()
                self.trees.append(f_tree)
            
            elif tree_type == 'master':
                ## adaptive tree
                if self.tree_design is not None:
                    # user-defined adaptive tree
                    m_tree = master_trees(data_tree, labels=labels_tree, 
                                                   child_slave_tree_params = self.tree_design)
                else:
                    # default use Kpotufe's adaptive tree
                    m_tree = master_trees(data_tree, labels=labels_tree)
                
                m_tree.build_master_trees()
                self.trees.append(m_tree)
            
            else:
                ## scikit implementation of DC tree
                pass
    
    
    def build_forest_with_tree_preproc(self, method):
        pass
        
    def build_forest_with_forest_preproc(self, method):
        pass
    
    
    def train(self):
        self.build_forest_classic(isPredict=True)
    
    def predict_one(self, point):
        """
        Predictor_type can be either 'class' for classification,
        'regress' for regression, or a user-defined callable function
        """
        assert self.trees, "You must first build a forest"
        
        if self.predictor_type == 'class':
            ## binary classification
            avg_predict = 0
            for tree in self.trees:
                avg_predict += tree.predict_one(point, predict_type='class')
            if avg_predict/float(len(self.trees)) > 0.5:
                return 1
            else:
                return 0
        elif self.predictor_type == 'regress':
            ## regression
            avg_predict = 0
            for tree in self.trees:
                avg_predict += tree.predict_one(point, predict_type='regress')
            return avg_predict/float(len(self.trees))
        else:
            print("Unrecognized prediction method!")
           
    def predict(self, test):
        predictions = list()
        for point in test:
            predictions.append(self.predict_one(point))
        return predictions