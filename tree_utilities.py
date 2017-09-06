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
#from pandas import DataFrame
#from IPython.display import display

####################
###---------- Utility functions
## computing spread of one-dim data
def spread_1D(data_1D):
    if len(data_1D)==0:
        return 0
    return np.amax(data_1D)-np.amin(data_1D)

## computing data diamter of a cell
def data_diameter(data):
    """
    Input data is assumed to be confined in the
    desired cell
    """
    dist, indx, indy = 0, 0, 0
    if data.shape[0] == 1:
        return dist, indx, indy
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            dist_new = np.linalg.norm(data[i,:]-data[j,:])
            if dist_new > dist:
                dist, indx, indy = dist_new, i, j
    return dist, indx, indy

###---------- compressive projection matrix designs
def comp_projmat(data, **kwargs):
    """
    returns a projection matrix
    Warning: the projection matrix returned can be either dense or sparse
    """
    namelist = ['breiman','ho','tomita', 'dasgupta']
    assert kwargs['name'] in namelist, "No such method for constructing projection matrix!"
    
    if kwargs['name'] == 'breiman':
        ## Breiman's Forest-IC and Forest-RC
        s = kwargs['sparsity']
        d = kwargs['target_dim']
        A = np.zeros((data.shape[1],d))
        ## sample sparsity-constrained A
        for i in range(d):
            ind = np.random.choice(data.shape[1], size=s, replace=False)
            if s == 1:
                A[ind,i] = 1
            else:
                for j in range(len(ind)):
                    A[ind[j], i] = np.random.uniform(-1,1)
    
    elif kwargs['name'] == 'ho':
        ## rotation forest
        d = kwargs['target_dim']
        ## find A by PCA
    elif kwargs['name'] == 'tomita':
        ## randomer forest
        d = kwargs['target_dim']
        ## sample sparse A via very sparse rp
        density = 1/(data.shape[1]**(1/2)) #default density value
        if 'density' in kwargs:
            if kwargs['density'] <= 1 and kwargs['density']>0:
                density = kwargs['density']
        
        transformer = random_projection.SparseRandomProjection(n_components=d, density=density)    
        transformer.fit(data)
        A = transformer.components_.copy()
        A = A.T ## A is SPARSE!
        
    else:
        ## dasgupta rp-tree 
        d = 1 # default to a random vector          
        if 'target_dim' in kwargs:
            d = kwargs['target_dim']
        n_features = data.shape[1]
        A = np.zeros((data.shape[1], d))
        # sample dense projection matrix
        for i in range(d):
            A[:,i] = np.random.normal(0, 1/np.sqrt(n_features), n_features)

    return A

#######-------split designs

## cart splits
def cart_split(data, proj_mat, labels=None, regress=False):
    # test for the best split feature and threshold on data CART criterion
    if sps.issparse(proj_mat):
        #data_trans = sps.csr_matrix.dot(data, proj_mat).squeeze()
        data_trans = proj_mat.T.dot(data.T).T.squeeze()
    else:
        data_trans = np.dot(data, proj_mat) # n-by-d

    score, ind, thres = -999, None, None
    if data_trans.ndim == 1:
        if not regress:
            #classification
            score, thres = cscore(data_trans, labels)
        else:
            #regression
            score, thres = rscore(data_trans, labels)
        w = proj_mat
    
    else:
       
        for i in range(proj_mat.shape[1]):
            if not regress:
                # classification
                score_new, thres_new = cscore(data_trans[:,i], labels)
            else:
                # regression
                score_new, thres_new = rscore(data_trans[:,i], labels)
            if score_new > score:
                score = score_new
                ind = i
                thres = thres_new
        w = proj_mat[:,ind]
    return score, w, thres

def cscore(data_1D, labels):
    ## cart classification criterion
    score, thres = -999, None
    if not list(labels):
        return score, thres
    n = len(labels)
    p1 = np.mean(labels)
    data_sorted, ind_sorted = np.sort(data_1D), np.argsort(data_1D)
    for i in range(1,n):
        cell_l = ind_sorted[:i]
        cell_r = ind_sorted[i:]
        split_val = data_sorted[i]
        if not list(labels[cell_l]) or not list(labels[cell_r]):
            #Do nothing if after either of the left and right labels are empty
            pass
        else:
            p1_l = np.mean(labels[cell_l])
            p1_r = np.mean(labels[cell_r])
            n_l = len(cell_l)
            score_new = p1*(1-p1) - (n_l/n)*(1-p1_l)*p1_l - (n-n_l)/n*(1-p1_r)*p1_r  
            if score_new > score:
                score = score_new
                thres = split_val
    return score, thres

def rscore(data_1D, labels):
    ## cart regression criterion
    score, thres = -999, None
    if not list(labels):
        return score, thres
    n = len(labels)
    ybar = np.mean(labels)
    data_sorted, ind_sorted = np.sort(data_1D), np.argsort(data_1D)
    
    for i in range(1,n):
        cell_l = ind_sorted[:i]
        cell_r = ind_sorted[i:]
        split_val = data_sorted[i]
        if not list(labels[cell_l]) or not list(labels[cell_r]):
            #Do nothing if after either of the left and right labels are empty
            pass
        else:
            ybar_l = np.mean(labels[cell_l])
            ybar_r = np.mean(labels[cell_r])
            score_new =(np.sum((labels-ybar)**2)-np.sum((labels[cell_l]-ybar_l)**2)\
                        -np.sum((labels[cell_r]-ybar_r)**2))/n
            if score_new > score:
                score = score_new
                thres = split_val
    return score, thres

##### median splits

def median_split(data, proj_mat, labels=None):
    if sps.issparse(proj_mat):
        #data_transformed = sps.csr_matrix.dot(data, proj_mat).squeeze()
        data_trans = proj_mat.T.dot(data.T).T.squeeze()
    else:
        data_trans = np.dot(data, proj_mat) # n-by-d
    if data_trans.ndim > 1:
        score, ind = 0, 0
        for i in range(proj_mat.shape[1]):
            score_new = spread_1D(data_trans[:,i])
            if score_new > score:
                score, ind = score_new, i
        w = proj_mat[:, ind]
        thres = np.median(data_trans[:,ind])
    else:
        thres = np.median(data_trans)
        w = proj_mat
        score = spread_1D(data_trans)
    return score, w, thres

def median_perturb_split(data, proj_mat, node_height, labels=None, diameter=None):

    if (node_height) % 2 == 0:
        # normal median splits
        return median_split(data, proj_mat, labels=labels)
    else:
        assert diameter is not None, "Diameter of the cell must be given!"
        # noisy splits
        if sps.issparse(proj_mat):
            #data_transformed = sps.csr_matrix.dot(data, proj_mat).squeeze()
            data_trans = proj_mat.T.dot(data.T).T.squeeze()
        else:
            data_trans = np.dot(data, proj_mat) # n-by-d
        
        if data_trans.ndim > 1:
            ## We use spread as the way to choose the best feature (could be changed)
            score, ind = 0, 0
            for i in range(proj_mat.shape[1]):
                score_new = spread_1D(data_trans[:, i])
                if score_new > score:
                    score, ind = score_new, i
            w = proj_mat[:, ind]
            jitter = np.random.uniform(-1,1) * 6/np.sqrt(data.shape[1]) * diameter
            thres = np.median(data_trans[:,ind])+jitter
        
        else:
            jitter = np.random.uniform(-1,1) * 6/np.sqrt(data.shape[1]) * diameter
            thres = np.median(data_trans)+jitter
            w = proj_mat
            score = spread_1D(data_trans)
        
    
        return score, w, thres

## 2-means split
def two_means_split(data, proj_mat, labels=None):
    # this essentially defines a hierarchical clustering on data
    return score, w, thres

#######-------stop rules
def naive_stop_rule(data, height=None):
    if data.shape[0] <= 1:
        return True
    if height > 8:
        ## DO NOT ever make it exceed 15!!!
        return True
    
    return False

def cell_size_rule(data, height, max_height=None, target_diameter=None):
    ddiameter,_,_ = data_diameter(data)
    if target_diameter < 0.01:
        return True
    if ddiameter <= target_diameter:
        return True
    elif max_height is not None and height > max_height:
        return True
    else:
        return False


