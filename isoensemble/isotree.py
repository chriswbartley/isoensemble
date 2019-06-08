# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:06:22 2016

@author: 19514733
"""
#from sklearn.tree._criterion import Criterion
#from sklearn.tree import _tree, _splitter, _criterion
#CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
import numpy as np
#from gen_iso_regn import GeneralisedIsotonicRegression
#from isoensemble import GeneralisedIsotonicRegression
import isoensemble
from sklearn.externals import six
import numbers
from sklearn.base import BaseEstimator
from itertools import combinations
#import matplotlib.pyplot as plt
from copy import deepcopy
#import warnings
#from sklearn.exceptions import DataConversionWarning as _DataConversionWarning
#from sklearn.exceptions import NonBLASDotWarning as _NonBLASDotWarning
from sklearn.utils.validation import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import compute_sample_weight
from sklearn.utils import check_random_state,check_array
from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
# from sklearn.isotonic import IsotonicRegression

import networkx as nx
from collections import defaultdict
import time     
from numpy import log
from math import erf

RULE_LOWER_CONST = -1e9
RULE_UPPER_CONST = 1e9

    # split_criterion values:
sc_dict=dict()
sc_dict['both_sides_have_min_sample_wgt']=3
sc_dict['both_sides_have_pts']=1
sc_dict['incomp_side_has_pts']=2
    # split_weight values:
sw_dict=dict()
sw_dict['univar_prob_distn'] =3
sw_dict['parent_weight'] = 0
sw_dict['contained_pts_weight'] = 1
sw_dict['hybrid_prob'] =2
sw_dict['hybrid_prob_empirical'] = 4
    
log2=lambda x:log(x)/log(2)                                             
verbose=False
def timeit(method):
 
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
 
        #print ('%r (%r, %r) %2.2f sec' ,(method.__name__, args, kw, te-ts)
        if verbose: 
            print('TIME - ' + method.__name__ + ': ' + str(te-ts) + ' secs')
        return result
 
    return timed
def remove_redundant_edges(G):
    processed_child_count = defaultdict(int)  #when all of a nodes children are processed, we'll add it to nodes_to_process
    descendants = defaultdict(set)            #all descendants of a node (including children)
    out_degree = {node:G.out_degree(node) for node in G.nodes_iter()}
    nodes_to_process = [node for node in G.nodes_iter() if out_degree[node]==0] #initially it's all nodes without children
    while nodes_to_process:
        next_nodes = []
        for node in nodes_to_process:
            '''when we enter this loop, the descendants of a node are known, except for direct children.'''
            for child in G.neighbors(node):
                if child in descendants[node]:  #if the child is already an indirect descendant, delete the edge
                    G.remove_edge(node,child)
                else:                                    #otherwise add it to the descendants
                    descendants[node].add(child)
            for predecessor in G.predecessors(node):             #update all parents' indirect descendants
                descendants[predecessor].update(descendants[node])  
                processed_child_count[predecessor]+=1            #we have processed one more child of this parent
                if processed_child_count[predecessor] == out_degree[predecessor]:  #if all children processed, add to list for next iteration.
                    next_nodes.append(predecessor)
        nodes_to_process=next_nodes
        

                            
# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
def entropy(probabilities):
   
   # Now calculate the entropy
   ent=0.0
   for p in probabilities:
      #p=float(results[r])/len(rows)
     if p!=0 and p!=1:
         ent=ent-p*log2(p)
   return ent
   
def intrinsic_value(partition_proportions): #also known as 'splitInfo'
    # Now calculate the entropy
   iv=0.0
   for p in partition_proportions:
     if p!=0 and p!=1:
         iv=iv-p*log2(p)
   return iv
   
# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
#def gini(probabilities):
#   #from math import log
#   #log2=lambda x:log(x)/log(2)  
#   results=uniquecounts(rows)
#   # Now calculate the class probabilities
#   p={}
#   for r in results.keys():
#      p[r]=float(results[r])/len(rows)
#   # calculate the gini
#   gini_res=0.0
#   for r1 in results.keys():
#       for r2 in results.keys():
#           if r1!=r2:
#               gini_res+=p[r1]*p[r2]
#   return gini_res
   

def gini_l1(probabilities):
    s=0
    for i in np.arange(len(probabilities)):
        for j in np.arange(len(probabilities)):
            if j!=i:
                s+=np.abs(i-j)*probabilities[i]*probabilities[j]
    return s/(len(probabilities)*(len(probabilities)-1))# sum(p*(1-p) for p in probabilities)
    
def gini(probabilities):
    return  sum(p*(1-p) for p in probabilities)

def extract_scikit_tree(scikit_decision_tree,output_tree):
    """Helper to turn a scikit decision tree into a 
    pure python tree of class DecisionTree
    """
    tree=scikit_decision_tree.tree_
    num_nodes = tree.node_count
#    leaf_ids = np.zeros([num_nodes], dtype=np.int32)
#    leaf_values = np.zeros([num_nodes], dtype=np.float64)
#    rule_upper_corners = np.ones(
#        [num_nodes, num_feats], dtype=np.float64) * np.inf
#    rule_lower_corners = np.ones(
#        [num_nodes, num_feats], dtype=np.float64) * -np.inf
    def traverse_nodes(node_id=0,output_node=None,
                       operator=None,
                       threshold=None,
                       feature=None,
                       new_lower=None,
                       new_upper=None):
        if output_node is None:
            output_node=output_tree.root_node
            
      
#        if node_id == 0:
#            new_lower = np.ones(num_feats) * RULE_LOWER_CONST
#            new_upper = np.ones(num_feats) * RULE_UPPER_CONST
#        else:
#            if operator == +1:
#                new_upper[feature] = threshold
#            else:
#                new_lower[feature] = threshold
        # tree.children_left[node_id] != tree.children_right[node_id]: #not
        # tree.feature[node_id] == -2:
        if tree.children_left[node_id] != TREE_LEAF:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            output_node.split()
            X=output_tree.train_X[output_node.train_data_idx]
            y=output_tree.train_y[output_node.train_data_idx]
            sample_weight=output_tree.sample_weight[output_node.train_data_idx]
            # transfer node characteristics
            output_node.decision_feat=feature+1
            output_node.decision_data_type=output_tree.feat_data_types[output_node.decision_feat-1]
            output_node.decision_values=threshold
            [indx_left, indx_right,split_pt_act]=output_tree.get_split_indxs(X,output_node.decision_feat,output_node.decision_values,output_node.decision_data_type)
            output_node.left.update_ys(y[indx_left],sample_weight[indx_left])
            output_node.right.update_ys(y[indx_right],sample_weight[indx_right])
            output_node.left.train_data_idx=output_node.train_data_idx[indx_left]
            output_node.right.train_data_idx=output_node.train_data_idx[indx_right]
            # calculate upper and lower corners of new nodes
            output_node.left.corner_lower=output_node.corner_lower.copy()
            output_node.left.corner_upper=output_node.corner_upper.copy()
            output_node.right.corner_lower=output_node.corner_lower.copy()
            output_node.right.corner_upper=output_node.corner_upper.copy()
            if output_node.decision_data_type=='ordinal':
                output_node.left.corner_upper[output_node.decision_feat-1]=output_node.decision_values
                output_node.right.corner_lower[output_node.decision_feat-1]=output_node.decision_values  
            

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id,output_node.left, +1, threshold, feature,
                           None,None)#new_lower.copy(), new_upper.copy())  # "<="

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id,output_node.right, -1, threshold, feature,
                           None,None)#new_lower.copy(), new_upper.copy())  # ">"
        else:  # a leaf node
            output_node.fuse()
            if node_id == 0: # the base node (0) is the only node!
                print('Warning: Tree only has one node! (i.e. the root node)')
            return None

    traverse_nodes()
    output_tree.number_nodes()
    
    return output_tree
        
CRITERIA_CLF = {'gini_l1': gini_l1,'gini': gini, 'entropy': entropy, 'info_gain_ratio': entropy}
   
class DecisionNode(object):
    def __init__(self, **kwargs):
        self.decision_feat=None if 'decision_feat' not in kwargs.keys() else kwargs['decision_feat']
        self.decision_data_type=None if 'decision_data_type' not in kwargs.keys() else kwargs['decision_data_type']
        self.decision_values=None if 'decision_value' not in kwargs.keys() else kwargs['decision_value']
        self._classes=None  if 'classes' not in kwargs.keys() else kwargs['classes']
        self._probabilities=None  if 'probabilities' not in kwargs.keys() else kwargs['probabilities']
        if 'criterion'  in kwargs.keys():
            self.criterion=kwargs['criterion']
        else:
            self.criterion=None
        self.parent=None if 'parent' not in kwargs.keys() else kwargs['parent']
        self.left=None if 'left' not in kwargs.keys() else kwargs['left']
        self.right=None if 'right' not in kwargs.keys() else kwargs['right']
        self.train_data_idx=None   if 'train_data_idx' not in kwargs.keys() else kwargs['train_data_idx']  
        #self.size=len(self.train_data_idx)
        self.sample_weight=None if 'sample_weight' not in kwargs.keys() else kwargs['sample_weight']
        # calculate various parameters
        if 'ys' in kwargs.keys():
            if self.sample_weight is None:
                self.sample_weight=np.ones(len(kwargs['ys']))
            self.update_ys(None if 'ys' not in kwargs.keys() else kwargs['ys'],self.sample_weight)   
        else:
            self.size=None if 'size' not in kwargs.keys() else kwargs['size']
        self.depth =0 if 'depth' not in kwargs.keys() else kwargs['depth']
        self.index=-1
        self.index_leaf=-1
        self.corner_upper=None if 'corner_upper' not in kwargs.keys() else kwargs['corner_upper']
        self.corner_lower=None if 'corner_lower' not in kwargs.keys() else kwargs['corner_lower']
        # temporary vars for debugging isoforest:
        self.estimators=None if 'estimators' not in kwargs.keys() else kwargs['estimators']
        self.estimators_probs=None if 'estimators_probs' not in kwargs.keys() else kwargs['estimators_probs']
        self.estimators_leaf_ids=None if 'estimators_leaf_ids' not in kwargs.keys() else kwargs['estimators_leaf_ids']
        self.path=[] if 'path' not in kwargs.keys() else kwargs['path']
        #self.resubst_err_node=0 # set in update_ys()
        self.resubst_err_branch=0
        self.num_leaves_branch=0
        self.alpha_crit=np.inf
    def update_ys(self, ys,sample_weight):  
        
        if not ys is None:
            if sample_weight is None:
                self.sample_weight=np.ones(len(np.ravel(ys)))
            else:
                self.sample_weight=sample_weight
            #if len(np.ravel(ys))==0 or ys is None:
            self.tally=self.get_tally(ys)
            #self.error=None 
            self.size=np.sum(self.sample_weight)
            self.resubst_err_node=1*(1-np.max(self.probabilities)) # default to assume p(t) is 1 (root node). Otherwise will be updated in grow_node
        #else:
        #    self.tally=self.get_tally(ys)
        #    self.size=len(np.ravel(ys))
            #self.error=np.sum(self.predicted_class!=ys) /self.size
            
    def is_leaf(self):
        return self.left is None
        
    def is_root_node(self):
        return self.parent is None
        
    @property
    def classes(self):
        if self._classes is None:
            if self.tally is None:
                return None
            else:
                return self.tally.keys()
        else:
            return self._classes
    @property
    def impurity(self):
        if self.criterion is None:
            return None
        else:
            return self.criterion(self.probabilities)    
            
    @property
    def probabilities(self):
        if self._probabilities is None: # use tally to calculate probabilities
            if self.tally is None:
                return None
            else:
                if self.size ==0:
                    ps=[]
                    for c in self.classes:
                        ps.append(1.0/len(self.classes))
                else:
                    ps=[]
                    siz=0.
                    for r in self.tally.keys():
                        ps.append(self.tally[r])
                        siz=siz+self.tally[r]
                    ps=list(np.asarray(ps,dtype='float')/siz)
                return ps
        else: # use overridden probabilities in _probabilities
            return self._probabilities
    @property
    def predicted_class(self):
        cum_prob=0.
        i_class=0
        probs=self.probabilities
        #res=None
        got_result=0
        for r in self.classes: # find lowest MEDIAN
            cum_prob=cum_prob+probs[i_class]
            if cum_prob>=0.5:
                res=r
                got_result=1
                break
            i_class=i_class+1
#        if res==None:
#           print(str(self.classes) + ' ' + str(self.probabilities))
        if got_result==0:
            print('not defined')
        #print(str(self.classes) + ' ' + str(self.probabilities) + ' ' + str(self.tally))    
        return res
#    def predicted_class(self):
#        max_prob=-99
#        max_class=-99
#        i_class=0
#        for r in self.classes:
#            if self.probabilities[i_class]>max_prob:
#                max_prob=self.probabilities[i_class]
#                max_class=r
#            i_class=i_class+1
#        return max_class
    def split(self):
        new_node_left=DecisionNode(parent=self,depth=self.depth+1,criterion=self.criterion,classes=self.classes)#,path=self.path+[self.index])
        new_node_right=DecisionNode(parent=self,depth=self.depth+1,criterion=self.criterion,classes=self.classes)#,path=self.path+[self.index])
        self.left=new_node_left
        self.right=new_node_right
        return
    def fuse(self):
        self.left=None
        self.right=None
        return

    def get_tally(self,ys):
       results={}
       for c in self.classes: 
           results[c]=0
       for iy in np.arange(len(ys)):
           y=ys[iy]
           wgt=self.sample_weight[iy]
           # The result is the last column
           if y not in results.keys(): results[y]=0
           results[y]+=wgt
       return results

#def populate_nodes_c(tree_features, 
#                         tree_thresholds, 
#                        tree_values,
#                        tree_left_children, 
#                       tree_right_children, 
#                       train_X,
#                       train_y,
#                       out_node_train_idxs, 
#                       out_node_train_nums)       :
#    #k_feat
#    #i
#    #j_node
#    n_samples=train_X.shape[0]
#    n_nodes=out_node_train_idxs.shape[0]
#    #n_pts_per_node=np.zeros(n_nodes,dtype=np.int32)
#    for i in np.arange(n_samples):
#        j_node=0
#        out_node_train_idxs[j_node,out_node_train_nums[j_node]]=i
#        out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
#        while tree_left_children[j_node] != TREE_LEAF: 
#            k_feat=tree_features[j_node]
#            if train_X[i,k_feat] <= tree_thresholds[j_node]: 
#                j_node=tree_left_children[j_node]
#            else:
#                j_node=tree_right_children[j_node]
#            out_node_train_idxs[j_node,out_node_train_nums[j_node]]=i
#            out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
#    return 

#def apply_c(tree_features, 
#                         tree_thresholds, 
#                        tree_values,
#                        tree_left_children, 
#                       tree_right_children, 
#                       train_X,
#                       out_node_idxs)       :
#    #k_feat
#    #i
#    #j_node
#    n_samples=train_X.shape[0]
#    #n_nodes=out_node_train_idxs.shape[0]
#    #n_pts_per_node=np.zeros(n_nodes,dtype=np.int32)
#    for i in np.arange(n_samples):
#        j_node=0
#        #out_node_train_idxs[j_node,out_node_train_nums[j_node]]=i
#        #out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
#        while tree_left_children[j_node] != TREE_LEAF: 
#            k_feat=tree_features[j_node]
#            if train_X[i,k_feat] <= tree_thresholds[j_node]: 
#                j_node=tree_left_children[j_node]
#            else:
#                j_node=tree_right_children[j_node]
#            #out_node_train_idxs[j_node,out_node_train_nums[j_node]]=i
#            #out_node_train_nums[j_node]=out_node_train_nums[j_node]+1
#        out_node_idxs[i]=j_node
#    return 

class LeafIDs(object):
    def __init__(self,bool_mask):
        self.capacity=len(bool_mask)
        self.leaf_index=np.zeros(self.capacity,dtype=np.int32)-1
        self.curr_size=np.sum(bool_mask)
        self.leaf_index[bool_mask]=np.arange(self.curr_size)
        self.leaf_array=np.zeros(self.capacity,dtype=np.int32)
        self.leaf_array[0:self.curr_size]=np.arange(self.capacity)[bool_mask]
      
    def fuse_branch(self,child_to_remove_1, child_to_remove_2,new_branch_leaf):
        self.leaf_array[self.leaf_index[child_to_remove_1]]=new_branch_leaf
        self.leaf_index[new_branch_leaf]=self.leaf_index[child_to_remove_1]
        self.leaf_index[child_to_remove_1]=-1
        self.remove_leaf(child_to_remove_2)
        
        
    def replace_leaf_with_chn(self,leaf_to_replace,new_leaf_1, new_leaf_2):
        self.leaf_array[self.leaf_index[leaf_to_replace]]=new_leaf_1
        self.leaf_index[new_leaf_1]=self.leaf_index[leaf_to_replace]
        self.leaf_index[leaf_to_replace]=-1
        self.leaf_array[self.curr_size]=new_leaf_2
        self.leaf_index[new_leaf_2]=self.curr_size
        self.curr_size=self.curr_size+1
        
    def get_list(self):
        return list(self.leaf_array[0:self.curr_size])

    def get_idx_array(self):
        return self.leaf_array[0:self.curr_size]

    
    def remove_leaf(self, leaf_to_remove):
        index=self.leaf_index[leaf_to_remove]
        if index<(self.curr_size-1):
            self.leaf_array[self.leaf_index[leaf_to_remove]]=self.leaf_array[self.curr_size-1]
            self.leaf_index[self.leaf_array[self.curr_size-1]]=self.leaf_index[leaf_to_remove]
        self.leaf_index[leaf_to_remove]=-1
        self.curr_size=self.curr_size-1            
        
class DecisionTreeArray(object):
    def __init__(self,sklearn_tree_,num_feats,num_classes,
                 incr_feats,decr_feats,
                 train_X,train_y,train_sample_weight,
                 allow_extra_nodes=0, 
                 normalise_nmt_nodes=0,split_criterion=None, 
                 split_class=None,split_weight=None,min_split_weight=0.5,
                 univariate_distns=None):
        self.num_nodes=sklearn_tree_.node_count
        #self.node_count=sklearn_tree_.node_count
        self.num_feats=num_feats
        self.num_classes=num_classes
        
        #num_classes=tree.value.shape[2]
        self.assumed_max_nodes=self.num_nodes+allow_extra_nodes
        self.features=np.zeros([self.assumed_max_nodes],dtype=np.int32)
        self.thresholds=np.zeros([self.assumed_max_nodes],dtype=np.float64)
        self.children_left=np.zeros([self.assumed_max_nodes],dtype=np.int32)
        self.children_right=np.zeros([self.assumed_max_nodes],dtype=np.int32)
        self.values=np.zeros([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.cdf_data=np.ones([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.cdf=np.ones([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.sample_weight=np.zeros([self.assumed_max_nodes],dtype=np.float64)
        self.pred_class=np.zeros([self.assumed_max_nodes],dtype=np.int32)
        
        self.features[0:self.num_nodes]=sklearn_tree_.feature
        self.thresholds[0:self.num_nodes]=sklearn_tree_.threshold
        self.children_left[0:self.num_nodes]=sklearn_tree_.children_left
        self.children_right[0:self.num_nodes]=sklearn_tree_.children_right
        self.values[0:self.num_nodes,:]=sklearn_tree_.value[:,0,:]
        self.sample_weight[0:self.num_nodes]=np.sum(self.values[0:self.num_nodes,:],axis=1)
        cum_sum=np.zeros(self.num_nodes,dtype=np.float64)
        for i_class in np.arange(num_classes-1):
            cum_sum=cum_sum+self.values[0:self.num_nodes,i_class]
            self.cdf_data[0:self.num_nodes,i_class]=cum_sum/self.sample_weight[0:self.num_nodes]    
        self.cdf=self.cdf_data.copy()
        self.pred_class[0:self.num_nodes]=np.argmax(self.cdf[0:self.num_nodes,:]>=0.5, axis=1)
        
        # get tree corners
        leaf_ids = np.zeros([self.assumed_max_nodes], dtype=np.int32)-99
        upper_corners = np.ones(
        [self.assumed_max_nodes, num_feats], dtype=np.float64,order='C') * RULE_UPPER_CONST
        lower_corners = np.ones(
        [self.assumed_max_nodes, num_feats], dtype=np.float64,order='C') * RULE_LOWER_CONST

        isoensemble.extract_rules_from_tree_c(sklearn_tree_.children_left.astype(np.int32),sklearn_tree_.children_right.astype(np.int32),sklearn_tree_.feature.astype(np.int32),sklearn_tree_.threshold.astype(np.float64), np.int32(num_feats), leaf_ids,upper_corners,lower_corners)
        self.upper_corners=upper_corners
        self.lower_corners=lower_corners
        self.leaf_ids_obj=LeafIDs(leaf_ids!=-99)# leaf_ids[leaf_ids!=-99]
        self.peak_leaves=self.leaf_ids_obj.curr_size  #len(self.leaf_ids)
        self.set_mt_feats(incr_feats,decr_feats)
        
        self.normalise_nmt_nodes=normalise_nmt_nodes
        self.split_criterion=split_criterion
        self.split_class=split_class
        self.split_weight=split_weight
        self.min_split_weight=min_split_weight
        
        # populate with training data
        self.train_X=train_X
        self.train_y=train_y
        self.train_sample_weight=train_sample_weight
        self.node_train_idx=np.zeros([self.assumed_max_nodes,train_X.shape[0]],dtype=np.int32)-99
        self.node_train_num=np.zeros([self.assumed_max_nodes],dtype=np.int32)
        isoensemble.populate_nodes_c(self.features, 
                         self.thresholds, 
                        self.values,
                        self.children_left, 
                       self.children_right, 
                       self.train_X.astype(np.float64),
                       self.train_y,
                       self.node_train_idx, 
                       self.node_train_num)
        
        self.univariate_distns=univariate_distns
        
        self.free_node_ids=np.zeros(1500,dtype=np.int32)
        self.free_node_ids_num=0
        
        self.free_node_ids_num_arr_=np.zeros(1,dtype=np.int32)
        self.num_nodes_arr_=np.zeros(1,dtype=np.int32)
        self.l_curr_size_=np.zeros(1,dtype=np.int32)
        self.done_normalising=False
        return
        
    def trim_to_size(self):
        self.free_node_ids=None
        self.free_node_ids_num=0
        self.train_X=None
        self.train_y=None
        self.node_train_idx=self.node_train_idx[0:self.num_nodes,:]
        self.node_train_num=self.node_train_num[0:self.num_nodes]
        self.features=self.features[0:self.num_nodes]
        self.thresholds=self.thresholds[0:self.num_nodes]
        self.children_left=self.children_left[0:self.num_nodes]#np.zeros([self.assumed_max_nodes],dtype=np.int32)
        self.children_right=self.children_right[0:self.num_nodes]#np.zeros([self.assumed_max_nodes],dtype=np.int32)
        self.values=self.values[0:self.num_nodes,:]#np.zeros([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.cdf_data=self.cdf_data[0:self.num_nodes,:]#np.ones([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.cdf=self.cdf[0:self.num_nodes,:]#np.ones([self.assumed_max_nodes,num_classes],dtype=np.float64)
        self.sample_weight=self.sample_weight[0:self.num_nodes]#np.zeros([self.assumed_max_nodes],dtype=np.float64)
        self.pred_class=self.pred_class[0:self.num_nodes]
        return
        
    def set_mt_feats(self,incr_feats,decr_feats):
        if incr_feats is not None or decr_feats is not None :
            self.in_feats=np.asarray(incr_feats)-1
            self.de_feats=np.asarray(decr_feats)-1
            mt_feats=list(self.in_feats).copy()
            for i in np.arange(len(self.de_feats)): mt_feats.append(self.de_feats[i])
            self.mt_feats=mt_feats
            self.nmt_feats=np.asarray([f for f in np.arange(self.num_feats) if f not in mt_feats])
            self.mt_feat_types=np.zeros(self.num_feats,dtype=np.int32)
            if len(self.in_feats)>0:
                self.mt_feat_types[self.in_feats]=+1
            if len(self.de_feats)>0:
                self.mt_feat_types[self.de_feats]=-1

    
    def get_increasing_leaf_node_pairs(self):
        #probs,lowers,uppers=self.get_corner_matrices()
        
        max_pairs=int(np.round(self.lower_corners.shape[0]*self.lower_corners.shape[0]))
        incr_pairs=np.zeros([max_pairs,2],dtype=np.int32)
        n_pairs_new=isoensemble.get_increasing_leaf_node_pairs_array(self.lower_corners,self.upper_corners,self.leaf_ids_obj.get_idx_array(),self.mt_feat_types,incr_pairs)
        #incr_pairs_old=self.get_increasing_leaf_node_pairs_simple()
        return incr_pairs[0:n_pairs_new,:]
        
    def eliminate_unnecessary_incr_pairs(self,pm_pairs):
#        G=nx.DiGraph()
#        #G.add_nodes_from(np.arange(np.max(self.leaf_ids_obj.get_idx_array())+1,dtype='int'))
#        G.add_edges_from(pm_pairs)
#        remove_redundant_edges(G)
#        res=G.edges().copy()
#        return res
        # faster cython immplementation
        # first reduce number of leaves
#        uniq_leaves=np.unique(pm_pairs)
#        lkp=dict()
#        rev_lkp=dict()
#        for i in np.arange(len(uniq_leaves)):
#            lkp[uniq_leaves[i]]=i
#            rev_lkp[i]=uniq_leaves[i]
#        pm_pairs_sequenced=np.zeros(pm_pairs.shape,dtype=np.int32)
#        for i in np.arange(pm_pairs.shape[0]):
#             pm_pairs_sequenced[i,0]= lkp[pm_pairs[i,0]]  
#             pm_pairs_sequenced[i,1]= lkp[pm_pairs[i,1]]  
        
        if len(pm_pairs)==0:
            return pm_pairs
        else:
            out_pm_pairs=np.zeros(pm_pairs.shape,dtype=np.int32)
            num_pairs=isoensemble.calculate_transitive_reduction_c(pm_pairs,out_pm_pairs)
            out_pm_pairs=out_pm_pairs[0:num_pairs,:]
            
            out_pm_pairs_w=np.zeros(pm_pairs.shape,dtype=np.int32)
            num_pairs=isoensemble.calculate_transitive_reduction_c_warshal(pm_pairs,out_pm_pairs_w)
            out_pm_pairs_w=out_pm_pairs_w[0:num_pairs,:]
            
            return out_pm_pairs
#        for i in np.arange(out_pm_pairs.shape[0]):
#            out_pm_pairs[i,0]= rev_lkp[out_pm_pairs[i,0]]  
#            out_pm_pairs[i,1]= rev_lkp[out_pm_pairs[i,1]] 
        
      
    def get_non_monotone_pairs(self,pm_pairs):
        nmt_pairs=[]
        for pair in pm_pairs:
            #if self.leaf_nodes[pair[0]].predicted_class>self.leaf_nodes[pair[1]].predicted_class:
            if self.pred_class[pair[0]]>self.pred_class[pair[1]]:
                nmt_pairs.append(pair)
        return nmt_pairs

    def clean_monotone_island_pairs(self,pm_pairs_clean,nmt_pairs):
        graph=nx.DiGraph()
        graph.add_edges_from(pm_pairs_clean) 
        ud_graph=graph.to_undirected()
        nodes_with_constraints =set(graph.nodes())
        unchecked_nodes=nodes_with_constraints.copy()
        polluted_nodes=set(np.unique(np.ravel(np.asarray(nmt_pairs))))
        safe_island_nodes_to_remove=[]#set()
        for n in  graph.nodes():
            if graph.predecessors(n) == []: # root node #successors(n)
                if n in unchecked_nodes:
                    nodes=set(nx.descendants(ud_graph,n))
                    has_no_nmt_polluted_nodes = len(nodes.intersection(polluted_nodes))==0
                    if has_no_nmt_polluted_nodes:
                        safe_island_nodes_to_remove=safe_island_nodes_to_remove+ list(nodes) + [n]
                    unchecked_nodes.difference_update(nodes)
                    unchecked_nodes.difference_update([n])
        cleaned_edges=nx.DiGraph()
        for edge in pm_pairs_clean:
            if edge[0] not in  safe_island_nodes_to_remove :
                cleaned_edges.add_edge(edge[0],edge[1])                         
        return cleaned_edges.edges()
    
    def get_next_free_node_ids(self,number=1):
        results=np.zeros(number,dtype=np.int32)
        #num_to_add=0
        for i in np.arange(number):
            idx=self.free_node_ids_num-1
            if idx>=0:
                results[i]=self.free_node_ids[idx]
                self.free_node_ids_num=self.free_node_ids_num-1
            else:
                results[i]=self.num_nodes#num_to_add
                #num_to_add=num_to_add+1
                self.num_nodes=self.num_nodes+1
        return results

    def return_free_node_id(self,node_id):
        #results=np.zeros(number,dtype=np.int32)
        #num_to_add=0
        self.free_node_ids[self.free_node_ids_num]=node_id
        self.free_node_ids_num=self.free_node_ids_num+1
        return
#        i=0
#        while self.children_left[i]!=0:
#            i=i+1
#        
#        first=i
#        i=i+1
#        while self.children_left[i]!=0:
#            i=i+1
       # return [first,i]
    def grow_segregated_nodes(self,node_to_grow,node_to_intersect_with):
        if self.split_weight not in ['hybrid_prob'  ,'prob_empirical_cond','hybrid_prob_empirical_orig_train' ]: #,'hybrid_prob_empirical'
            self.free_node_ids_num_arr_[0]=self.free_node_ids_num
            self.num_nodes_arr_[0]=self.num_nodes
            self.l_curr_size_[0]=self.leaf_ids_obj.curr_size
            
            change_made=isoensemble.grow_segregated_nodes_c(node_to_grow,
                    node_to_intersect_with,
                    self.free_node_ids, 
                    self.free_node_ids_num_arr_,
                    self.num_nodes_arr_, 
                    sc_dict[self.split_criterion],
                    self.sample_weight,
                    self.min_split_weight,
                    sw_dict[self.split_weight],
                    self.lower_corners,
                    self.upper_corners,
                    self.node_train_num,
                    self.node_train_idx,
                    self.assumed_max_nodes,
                    self.train_X,
                    self.train_y,
                    self.num_classes,
                    self.train_sample_weight,
                    self.features,
                    self.thresholds,
                    self.children_left,
                    self.children_right,
                    self.values,
                    self.cdf_data,
                    self.cdf,
                    self.pred_class,
                    self.leaf_ids_obj.leaf_index,
                    self.leaf_ids_obj.leaf_array,
                    self.l_curr_size_,
                    self.univar_vals,
                    self.univar_probs,
                    self.univar_vals_num
                    )==1
            # restore values passed by proxy array
            self.free_node_ids_num=self.free_node_ids_num_arr_[0]
            self.num_nodes=self.num_nodes_arr_[0]
            self.leaf_ids_obj.curr_size=self.l_curr_size_[0]
            # refresh leaf ids object
            #self.leaf_ids_obj=LeafIDs(self.children_left==TREE_LEAF)
        else: # old way
            #for node_to_grow,l2 in [[leafnode1,leafnode2],[leafnode2,leafnode1]]:
            l1=node_to_grow
            l2=node_to_intersect_with
            change_made=False
            [temp_split_left_node_id,temp_split_right_node_id]=self.get_next_free_node_ids(2)
            split_decision=False
            if self.split_criterion=='both_sides_have_min_sample_wgt' and self.sample_weight[l1]<2*self.min_split_weight: # there is no way to split this node, stop
                pass
            else:
                #feats=list(self.nmt_feats) + list( self.mt_feats) # should result in less splits (and nodes) if we split on NMT feats first
                for i_feat in np.arange(self.num_feats): #feats: # np.arange(len(l1.corner_lower)): #self.mt_feats:#  np.arange(len(l1.corner_lower)):
                    for dirn in ['left','right']:
                        split_val=-99e9
                        if self.split_weight!='univar_prob_distn' or (self.split_weight=='univar_prob_distn' and l1.size>0.000005): # don't split when it gets too small!!
                            if dirn=='right':
                                #if l1.corner_lower[i_feat]<l2.corner_lower[i_feat] and l1.corner_upper[i_feat]>l2.corner_lower[i_feat] : # slice off bottom bit
                                if self.lower_corners[l1,i_feat]<self.lower_corners[l2,i_feat] and self.upper_corners[l1,i_feat]>self.lower_corners[l2,i_feat] : # slice off bottom bit
                                    split_val=self.lower_corners[l2,i_feat]
                            else: # left
                                #if l1.corner_upper[i_feat]>l2.corner_upper[i_feat] and l1.corner_lower[i_feat]<l2.corner_upper[i_feat] :
                                if self.upper_corners[l1,i_feat]>self.upper_corners[l2,i_feat] and self.lower_corners[l1,i_feat]<self.upper_corners[l2,i_feat] :
                                    split_val=self.upper_corners[l2,i_feat]
                        if split_val!=-99e9: # need to split on this feat value
                            # work out which points go where for this proposed split
                            #temp_split_left_node_id=self.num_nodes
                            #temp_split_right_node_id=self.num_nodes+1
                            self.node_train_num[temp_split_left_node_id]=0
                            self.node_train_num[temp_split_right_node_id]=0
                            self.sample_weight[temp_split_left_node_id]=0.
                            self.sample_weight[temp_split_right_node_id]=0.
                            for i_ in np.arange(self.node_train_num[l1]):
                                i=self.node_train_idx[l1,i_]
                                if self.train_X[i,i_feat]<=split_val:
                                    self.node_train_idx[temp_split_left_node_id,self.node_train_num[temp_split_left_node_id]]=i
                                    self.node_train_num[temp_split_left_node_id]=self.node_train_num[temp_split_left_node_id]+1
                                    self.sample_weight[temp_split_left_node_id]=self.sample_weight[temp_split_left_node_id]+self.train_sample_weight[i]
                                else:
                                    self.node_train_idx[temp_split_right_node_id,self.node_train_num[temp_split_right_node_id]]=i
                                    self.node_train_num[temp_split_right_node_id]=self.node_train_num[temp_split_right_node_id]+1
                                    self.sample_weight[temp_split_right_node_id]=self.sample_weight[temp_split_right_node_id]+self.train_sample_weight[i]
                            # adjust child sample weights if required
                            if self.split_weight=='parent_weight':
                                self.sample_weight[temp_split_left_node_id]=self.sample_weight[l1]
                                self.sample_weight[temp_split_right_node_id]=self.sample_weight[l1]
                            elif self.split_weight=='contained_pts_weight':
                                pass # sample weights already correctly set
                                #self.sample_weight[temp_split_left_node_id]=self.sample_weight[temp_split_left_node_id]#np.max([0.5,self.sample_weight[temp_split_left_node_id]])
                                #self.sample_weight[temp_split_right_node_id]=self.sample_weight[temp_split_right_node_id]#np.max([0.5,self.sample_weight[temp_split_right_node_id]])
                            elif self.split_weight=='hybrid_prob' or self.split_weight=='hybrid_prob_empirical' or self.split_weight=='prob_empirical_cond' or self.split_weight=='hybrid_prob_empirical_orig_train' :
                                if self.sample_weight[temp_split_left_node_id]==0. or self.sample_weight[temp_split_right_node_id]==0.:
                                    if self.split_weight=='prob_empirical_cond':
                                        #[dist_vals,dist_probs]=self.univariate_distns[l1.predicted_class][i_feat]
                                        raise NotImplemented
                                    else:
                                        [dist_vals,dist_probs]=self.univariate_distns[i_feat]
                                    left_extents=[self.lower_corners[l1,i_feat],split_val]
                                    right_extents=[split_val,self.upper_corners[l1,i_feat]]
                                    
                                    #left_extents=[l1.corner_lower[i_feat],split_val]
                                    #right_extents=[split_val,l1.corner_upper[i_feat]]
                                    prob_left=calc_probability(dist_vals,dist_probs,left_extents[0],left_extents[1])
                                    prob_right=calc_probability(dist_vals,dist_probs,right_extents[0],right_extents[1])
                                    self.sample_weight[temp_split_left_node_id]=self.sample_weight[l1]*prob_left/(prob_left+prob_right)
                                    self.sample_weight[temp_split_right_node_id]=self.sample_weight[l1]*prob_right/(prob_left+prob_right)
                            elif self.split_weight=='univar_prob_distn':
                                raise NotImplemented
                            # make decision to split or not
                            if self.split_criterion=='both_sides_have_pts':
                                split_decision=self.sample_weight[temp_split_left_node_id]>0 and self.sample_weight[temp_split_right_node_id]>0
                            elif self.split_criterion=='incomp_side_has_pts' : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                                if dirn=='right':
                                    split_decision=self.sample_weight[temp_split_left_node_id]>0 
                                else: # left
                                    split_decision=self.sample_weight[temp_split_right_node_id]>0 
                            elif self.split_criterion=='both_sides_have_min_sample_wgt' : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                                split_decision=self.sample_weight[temp_split_left_node_id]>=self.min_split_weight and self.sample_weight[temp_split_right_node_id]>=self.min_split_weight
                            else: #if self.split_criterion=='all_splits_ok' : 
                                split_decision=True
                            # split if so decided
                            if split_decision:
                                change_made=True
                                self.features[l1]=i_feat
                                self.thresholds[l1]=split_val
                                self.children_left[l1]=temp_split_left_node_id
                                self.children_right[l1]=temp_split_right_node_id
                                self.children_left[temp_split_left_node_id]=TREE_LEAF
                                self.children_right[temp_split_left_node_id]=TREE_LEAF
                                self.children_left[temp_split_right_node_id]=TREE_LEAF
                                self.children_right[temp_split_right_node_id]=TREE_LEAF
                                
                                #self.num_nodes=np.max([temp_split_left_node_id+1,temp_split_right_node_id+1,self.num_nodes])
                                #self.free_node_ids[self.free_node_ids_num-1]=-99
                                #self.free_node_ids[self.free_node_ids_num-2]=-99
                                #self.free_node_ids_num=np.max([0,self.free_node_ids_num-2
                                
                                for i_ in np.arange(self.node_train_num[temp_split_left_node_id]):
                                    i=self.node_train_idx[temp_split_left_node_id,i_]
                                    self.values[temp_split_left_node_id,self.train_y[i]]=self.values[temp_split_left_node_id,self.train_y[i]]+self.train_sample_weight[i]
                                
                                for i_ in np.arange(self.node_train_num[temp_split_right_node_id]):
                                    i=self.node_train_idx[temp_split_right_node_id,i_]
                                    self.values[temp_split_right_node_id,self.train_y[i]]=self.values[temp_split_right_node_id,self.train_y[i]]+self.train_sample_weight[i]
                                
                                #self.values=np.zeros([self.assumed_max_nodes,num_classes],dtype=np.float64)
                                cum_sum=0.
                                for i_class in np.arange(self.num_classes):
                                    cum_sum=cum_sum+self.values[temp_split_left_node_id,i_class]
                                    self.cdf_data[temp_split_left_node_id,i_class]=cum_sum   
                                if cum_sum>0:
                                    self.cdf_data[temp_split_left_node_id,:]=self.cdf_data[temp_split_left_node_id,:]/cum_sum
                                
                                cum_sum=0.
                                for i_class in np.arange(self.num_classes):
                                    cum_sum=cum_sum+self.values[temp_split_right_node_id,i_class]
                                    self.cdf_data[temp_split_right_node_id,i_class]=cum_sum   
                                if cum_sum>0:
                                    self.cdf_data[temp_split_right_node_id,:]=self.cdf_data[temp_split_right_node_id,:]/cum_sum
                                
                                self.cdf[temp_split_left_node_id,:]=self.cdf[l1,:]#.copy()
                                self.cdf[temp_split_right_node_id,:]=self.cdf[l1,:]#.copy()
                                #self.sample_weight=np.zeros([self.assumed_max_nodes],dtype=np.float64)
                                self.pred_class[temp_split_left_node_id]=self.pred_class[l1]#np.argmax(self.cdf[temp_split_left_node_id,:]>=0.5, axis=0)
                                self.pred_class[temp_split_right_node_id]=self.pred_class[l1]#np.argmax(self.cdf[temp_split_right_node_id,:]>=0.5, axis=0)
                                
                                self.lower_corners[temp_split_left_node_id,:]=self.lower_corners[l1,:]
                                self.lower_corners[temp_split_right_node_id,:]=self.lower_corners[l1,:]
                                self.upper_corners[temp_split_left_node_id,:]=self.upper_corners[l1,:]
                                self.upper_corners[temp_split_right_node_id,:]=self.upper_corners[l1,:]
                                self.upper_corners[temp_split_left_node_id,i_feat]=split_val
                                self.lower_corners[temp_split_right_node_id,i_feat]=split_val
                                self.leaf_ids_obj.replace_leaf_with_chn(l1,temp_split_left_node_id,temp_split_right_node_id)
                                #leaf_ids_=list(self.leaf_ids)
                                #leaf_ids_.remove(l1)
                                #leaf_ids_=leaf_ids_+[temp_split_left_node_id,temp_split_right_node_id]
                                #self.leaf_ids=np.asarray(leaf_ids_)
                                #l1.left.resubst_err_node=l1.resubst_err_node #np.sum(self.sample_weight[indx_left])/self.sample_weight_total*(1-np.max(node.left.probabilities)) 
                                #l1.right.resubst_err_node=l1.resubst_err_node #np.sum(self.sample_weight[indx_right])/self.sample_weight_total*(1-np.max(node.right.probabilities)) 
                                # move to child to follow
                                [temp_split_left_node_id,temp_split_right_node_id]=self.get_next_free_node_ids(2)
                                #print(self.num_nodes)
                                l1=self.children_left[l1] if dirn=='left' else self.children_right[l1] 
            self.return_free_node_id(temp_split_left_node_id)
            self.return_free_node_id(temp_split_right_node_id)
        # update peak leaves if required
        num_leaves=self.leaf_ids_obj.curr_size#len(self.leaf_ids)
        if num_leaves>self.peak_leaves:
            self.peak_leaves=num_leaves
        return change_made
   
    def get_pairs_to_split(self,pm_pairs_clean,nmt_pairs,normalise_nmt_nodes):
        if self.normalise_nmt_nodes==0:
            pairs_to_split=[] #if normalise_nmt_nodes==1 else pm_pairs_clean
        elif self.normalise_nmt_nodes==1:
            pairs_to_split=nmt_pairs 
        elif self.normalise_nmt_nodes==2:
            pairs_to_split=pm_pairs_clean
        elif self.normalise_nmt_nodes==3:
            raise NotImplemented()
        return pairs_to_split
    
    def get_leaf_id_pairs(self,cleaned_pairs):
#        lookup=np.zeros(np.max(self.leaf_ids_obj.get_idx_array())+1,dtype=np.int32)
#        
#        l_=0
#        for l in self.leaf_ids_obj.get_idx_array():
#            lookup[l]=l_
#            l_=l_+1
#        leaf_id_pairs=[]
#        for pair in cleaned_pairs:
#            leaf_id_pairs=leaf_id_pairs+[(lookup[pair[0]],lookup[pair[1]])]        
        #return leaf_id_pairs   
        pairs=np.asarray(cleaned_pairs,dtype=np.int32)         
        
        if pairs.shape[0]>0:
            out_seq_pairs=np.zeros(pairs.shape,dtype=np.int32)
            leaf_ids=self.leaf_ids_obj.get_idx_array()
            isoensemble.get_leaf_id_pairs_c(pairs,leaf_ids,out_seq_pairs)
        return out_seq_pairs
    
#    def recalculate_leaf_ids(self):
#        leaf_ids=np.zeros(2000,dtype=np.int32)
#        num_leaves=0
#        for node_id in np.arange(self.num_nodes):
#            if self.children_left[node_id]==TREE_LEAF:
#                leaf_ids[num_leaves]=node_id
#                num_leaves=num_leaves+1
#        self.leaf_ids=leaf_ids[0:num_leaves]
#        max_leaf_id=self.num_nodes
#        while self.children_left[max_leaf_id]==0:
#            max_leaf_id=max_leaf_id-1
#        #self.num_nodes=max_leaf_id
#        if num_leaves>self.peak_leaves:
#            self.peak_leaves=num_leaves
#        #print(self.num_nodes)
            
    # find any pairse of child nodes with a common parent and the same predicted class. If found, fuse the parent.
    # returns 0 if no nodes were fused, or else the number of nodes fused.
    def simplify_old(self,node_id=None):
        if node_id is None: node_id=0
        res=self.simplify_recurse(node_id)
        self.recalculate_leaf_ids()
        return res
    
    def simplify_recurse(self,node_id=None):
        if node_id is None: node_id=0
        if self.children_left[node_id]==TREE_LEAF:
            return 0
        else:
            left_id=self.children_left[node_id]
            right_id=self.children_right[node_id]
            if self.children_left[left_id]==TREE_LEAF and self.children_left[right_id]==TREE_LEAF: # this is an immediate parent node
                if self.pred_class[left_id]==self.pred_class[right_id]:
                #if node.left.predicted_class==node.right.predicted_class:
#                    if self.pred_class[node_id]!=elf.pred_class[left_id]: # these were obviously changed. set probabilities of node to average of children
#                        node._probabilities=list((np.asarray(node.left.probabilities)*node.left.size+np.asarray(node.right.probabilities)*node.right.size)/(node.left.size+node.right.size))
#                        if np.isnan(node._probabilities[0]):
#                            print('what the')
#                    node.fuse()
#                    self.number_nodes()
                    self.children_left[node_id]=TREE_LEAF 
                    self.children_right[node_id]=TREE_LEAF 
                    self.children_left[left_id]=0#TREE_LEAF 
                    self.children_left[right_id]=0#TREE_LEAF 
                    self.children_right[left_id]=0
                    self.children_right[right_id]=0
                    self.free_node_ids[self.free_node_ids_num]=left_id
                    self.free_node_ids_num=self.free_node_ids_num+1
                    self.free_node_ids[self.free_node_ids_num]=right_id
                    self.free_node_ids_num=self.free_node_ids_num+1
#                    leaf_ids_=list(self.leaf_ids)
#                    leaf_ids_.remove(left_id)
#                    leaf_ids_.remove(right_id)
#                    leaf_ids_=leaf_ids_+[node_id]
#                    self.leaf_ids=np.asarray(leaf_ids_)
                    return 1
                else:
                    return 0
            else: # node has children with different
                res_left=self.simplify_recurse(self.children_left[node_id])
                res_right=self.simplify_recurse(self.children_right[node_id])
                return res_left+res_right

    # find any pairse of child nodes with a common parent and the same predicted class. If found, fuse the parent.
    # returns 0 if no nodes were fused, or else the number of nodes fused.
    def simplify(self):
        num_fuses=0
        for node_id in np.arange(self.num_nodes):
            if self.children_left[node_id]!=  TREE_LEAF and  self.children_left[node_id]!=  0:
                left_id=self.children_left[node_id]
                right_id=self.children_right[node_id]
                if self.children_left[left_id]==TREE_LEAF and self.children_left[right_id]==TREE_LEAF: # this is an immediate parent node
                    if self.pred_class[left_id]==self.pred_class[right_id]:
                        self.children_left[node_id]=TREE_LEAF 
                        self.children_right[node_id]=TREE_LEAF 
                        self.children_left[left_id]=0#TREE_LEAF 
                        self.children_left[right_id]=0#TREE_LEAF 
                        self.children_right[left_id]=0
                        self.children_right[right_id]=0
                        self.leaf_ids_obj.fuse_branch(left_id,right_id,node_id)
                        self.free_node_ids[self.free_node_ids_num]=left_id
                        self.free_node_ids_num=self.free_node_ids_num+1
                        self.free_node_ids[self.free_node_ids_num]=right_id
                        self.free_node_ids_num=self.free_node_ids_num+1
                        num_fuses=num_fuses+1
        #self.recalculate_leaf_ids()
        return num_fuses
           
            
    def monotonise(self,sample_reweights=None,univar_vals=None,univar_probs=None,univar_vals_num=None):
        self.univar_vals=univar_vals
        self.univar_probs=univar_probs
        self.univar_vals_num=univar_vals_num
        #self.univariate_distns=univariate_distns
        # get increasing pairs
        pm_pairs=self.get_increasing_leaf_node_pairs()   
        pm_pairs_clean=self.eliminate_unnecessary_incr_pairs(pm_pairs)
        nmt_pairs=self.get_non_monotone_pairs(pm_pairs_clean)

        # decide on pair to split
        pairs_to_split=self.get_pairs_to_split(pm_pairs_clean,nmt_pairs,self.normalise_nmt_nodes)
            #pairs_to_split=self.get_non_monotone_pairs_extended(nmt_pairs,pm_pairs_clean)
        # stop if already monotone
        if nmt_pairs==[]: # already monotone
            return 0 
        # grow segregated nodes if requested
        if self.normalise_nmt_nodes>0 and not self.done_normalising:
            self.done_normalising=True
            keep_going=True
            while keep_going:
                keep_going=False
                change_made=False
                changed_nodes=[]
                #pairs_to_split=nmt_pairs if normalise_nmt_nodes==1 else pm_pairs_clean
                for pair in pairs_to_split:
                    if  pair[0] not in changed_nodes and pair[1] not in changed_nodes : # this filter seems a tad slower and hence complexity is not warranted
                        change_made1=self.grow_segregated_nodes(pair[0],pair[1])
                        change_made2=self.grow_segregated_nodes(pair[1],pair[0])
                        if change_made1: changed_nodes.append(pair[0])
                        if change_made2: changed_nodes.append(pair[1])
                        change_made=change_made or change_made1 or change_made2
                        

                if change_made:
                    #self.number_nodes()
                    #print(' now: ' + str(len(self.leaf_nodes)))
                    pm_pairs=self.get_increasing_leaf_node_pairs()   
                    pm_pairs_clean=self.eliminate_unnecessary_incr_pairs(pm_pairs)
                    nmt_pairs=self.get_non_monotone_pairs(pm_pairs_clean)
                    
                    pairs_to_split=self.get_pairs_to_split(pm_pairs_clean,nmt_pairs,self.normalise_nmt_nodes) #nmt_pairs if normalise_nmt_nodes==1 else pm_pairs_clean
                    keep_going=True
        # monotonise the cdf
        cleaned_pairs=self.clean_monotone_island_pairs(pm_pairs_clean,nmt_pairs)
        if sample_reweights is not None:
            raise NotImplemented
#            weights=self.get_leaf_sizes()
#        else:
#            weights=self.recalc_leaf_sizes(sample_reweights)

        leaf_id_pairs=self.get_leaf_id_pairs(cleaned_pairs)
        leaf_ids_=self.leaf_ids_obj.get_idx_array()
        cdf=self.cdf[leaf_ids_,:]   #get_cum_probabilities()
        weights=self.sample_weight[leaf_ids_]
        cdf_iso=np.ones(cdf.shape) 
        pdf_iso=np.zeros(cdf.shape) 
        cum_sse=0.          
        for i_class in np.arange(cdf.shape[1]):
            probs_class=cdf[:,i_class]
            gir=isoensemble.GeneralisedIsotonicRegression()
            if i_class<cdf.shape[1]-1:
                #cdf_iso[:,i_class]=gir.fit(probs_class,pm_pairs_clean,sample_weight=weights,increasing=False)
                #print(probs_class)
                cdf_iso[:,i_class]=np.round(gir.fit(probs_class,leaf_id_pairs,sample_weight=weights,increasing=False),6)
            if i_class==0:
                pdf_iso[:,i_class]=cdf_iso[:,i_class]
            else:
                pdf_iso[:,i_class]=cdf_iso[:,i_class]-cdf_iso[:,i_class-1]
        cum_sse=np.sum((cdf_iso-cdf)**2)
        # update the leave probabilities
        if cum_sse>1e-7: # some changes were made
            self.cdf[leaf_ids_,:]=cdf_iso
            self.pred_class[leaf_ids_]=np.argmax(self.cdf[leaf_ids_,:]>=0.5, axis=1)
#            for leaf in self.leaf_nodes:
#                leaf._probabilities=list(pdf_iso[leaf.index_leaf,:])
#                if np.isnan(leaf._probabilities[0]):
#                    print('what the')
            res= 1
        else: # effectively no changes made
            res= 0
        # check we now have monotone tree:
        nmt_pairs2=self.get_non_monotone_pairs(pm_pairs_clean)
        if len(nmt_pairs2)>0:
            print('ERROR: orig nmt pairs:' + str(len(nmt_pairs)) + " now: " + str(len(nmt_pairs2)))
        return res
    def apply(self,X):
        node_idxs=np.zeros([X.shape[0]],dtype=np.int32)
        isoensemble.apply_c(self.features, 
                         self.thresholds, 
                        self.values,
                        self.children_left, 
                       self.children_right, 
                       X.astype(np.float64,order='C'),
                       node_idxs)
        return node_idxs
    def predict_proba(self,X):
        node_idxs=self.apply(X)
        cum_probs_=self.cdf[node_idxs,:]
        probs_=cum_probs_.copy()#np.zeros(cum_probs_.shape,dtype=np.float64)

        for i in np.arange(1,probs_.shape[1]):
            probs_[:,i]=cum_probs_[:,i]-cum_probs_[:,i-1]
            
        #[ileaf,probs_,paths]=self.apply_base(X)
        return probs_

    def predict_cum_proba(self,X):
        node_idxs=self.apply(X)
        probs_=self.cdf[node_idxs,:]
        #[ileaf,probs_,paths]=self.apply_base(X)
        return probs_
            
class DecisionTree(object):
    def __init__(self,train_X, train_y,criterion,feat_labels='auto',feat_data_types='auto',split_pts=None,classes=None,sample_weight=None,incr_feats=None,decr_feats=None,split_criterion=None, split_class=None,split_weight=None,min_split_weight=1.,univariate_distns=None):
        self.n_features = train_X.shape[1]
        
        #self.nodes=[]
        self._feat_data_types=feat_data_types
        self._feat_labels=feat_labels
        self._feat_vals=None
        self.split_criterion=split_criterion
        self.split_class=split_class
        self.split_weight=split_weight
        self.min_split_weight=min_split_weight
        self.univariate_distns=univariate_distns
        if sample_weight is None : # B_TEST 1/2
            self.train_X=train_X
            self.train_y=train_y
            self.sample_weight=np.ones(len(train_y))
        else:
            self.sample_weight=sample_weight[sample_weight!=0.]
            self.train_X=train_X[sample_weight!=0.]
            self.train_y=train_y[sample_weight!=0.]
        self.classes=classes if classes is not None else np.sort(np.unique(self.train_y))
        self.n_classes = len(self.classes)
        self.sample_weight_total=np.sum(self.sample_weight)
        self.criterion=CRITERIA_CLF[criterion]
        self.leaf_nodes=[]
        self.branch_nodes=[]
        self.set_mt_feats(incr_feats,decr_feats)
        
        # append root node
        self.root_node=DecisionNode(train_data_idx=np.arange(self.train_X.shape[0]),ys=self.train_y,depth=0,criterion=self.criterion,classes=self.classes,sample_weight=self.sample_weight)
        # update corner nodes
        lower=[]
        upper=[]
        for ifeat in np.arange(self.train_X.shape[1])+1:                
            if self.feat_data_types[ifeat-1]=='ordinal':
                lower.append(-np.inf ) #[-np.inf for i in np.arange(self.n_features)]   
                upper.append(np.inf ) #=[np.inf for i in np.arange(self.n_features)]            
            else: #'nominal'
                lower.append(tuple(self.feat_vals[ifeat-1]))
                upper.append(tuple(self.feat_vals[ifeat-1]))
        self.root_node.corner_lower=lower
        self.root_node.corner_upper=upper
        self.leaf_nodes.append(self.root_node)   
        self.split_pts_= split_pts # if 'split_pts' not in kwargs.keys() else kwargs['split_pts']
        self.size=np.sum(self.sample_weight)
        self.impurity=self.root_node.impurity
        self.num_nmt_pairs=0.
        self.num_iterations=0.
        self.peak_leaves=0.

    def set_mt_feats(self,incr_feats,decr_feats):
        if incr_feats is not None or decr_feats is not None :
            self.in_feats=np.asarray(incr_feats)-1
            self.de_feats=np.asarray(decr_feats)-1
            mt_feats=list(self.in_feats).copy()
            for i in np.arange(len(self.de_feats)): mt_feats.append(self.de_feats[i])
            self.mt_feats=mt_feats
            self.nmt_feats=np.asarray([f for f in np.arange(self.n_features) if f not in mt_feats])
            self.mt_feat_types=np.zeros(self.train_X.shape[1],dtype=np.int32)
            if len(self.in_feats)>0:
                self.mt_feat_types[self.in_feats]=+1
            if len(self.de_feats)>0:
                self.mt_feat_types[self.de_feats]=-1
    @property
    def split_pts(self):
        if self.split_pts_ is None and not self.train_X  is None:
            self.split_pts_=[]
            for ifeat in np.arange(self.n_features)+1:
                pts=np.sort(np.unique(self.train_X[:,ifeat-1]))
                # NEW S_TEST 1/3
                self.split_pts_.append((pts[0:-1]+pts[1:])/2.)
                # OLD S_TEST
                #self.split_pts_.append(np.sort(np.unique(self.train_X[:,ifeat-1])))
                #split_vals=self.split_pts[ifeat-1]#(split_vals[1:len(split_vals)]+split_vals[0:len(split_vals)-1])/2.0             
        return self.split_pts_
        
#    def reweight_nodes(self, X_reweight,y_reweight=None,node=None):
#        if node is None: 
#            node=self.root_node
#            node.train_data_idx=np.arange(X_reweight.shape[0])
#        # reweight this node
#        if not y_reweight is None:
#            node.update_ys(y_reweight) # takes care of size, tally, and probabilities
#        else: # just reweight size
#            node._probabilities=node.probabilities #freeze probabilities to current value (before re-weight)
#            node.size=X_reweight.shape[0]            
#        # proceed on to children
#        if node.is_leaf():
#            return
#        else: # has children
#            [indx_left, indx_right]=self.get_split_indxs(X_reweight,node.decision_feat,node.decision_values,node.decision_data_type)
#            node.left.train_data_idx=node.train_data_idx[indx_left]
#            node.right.train_data_idx=node.train_data_idx[indx_right]
#            if y_reweight is None:
#                self.reweight_nodes(X_reweight[indx_left,:],None,node.left)   
#                return self.reweight_nodes(X_reweight[indx_right,:],None,node.right) 
#            else:
#                self.reweight_nodes(X_reweight[indx_left,:],y_reweight[indx_left],node.left)   
#                return self.reweight_nodes(X_reweight[indx_right,:],y_reweight[indx_right],node.right) 
#            
        
    def copy(self):
        return deepcopy(self)
        
    def grow_segregated_nodes(self,node_to_grow,node_to_intersect_with):
        #for node_to_grow,l2 in [[leafnode1,leafnode2],[leafnode2,leafnode1]]:
        l1=node_to_grow
        l2=node_to_intersect_with
        change_made=False
        if self.split_criterion=='both_sides_have_min_sample_wgt' and node_to_grow.size<2*self.min_split_weight: # there is no way to split this node, stop
            return False
        #feats=list(self.nmt_feats) + list( self.mt_feats) # should result in less splits (and nodes) if we split on NMT feats first
        for i_feat in np.arange(len(l1.corner_lower)): #feats: # np.arange(len(l1.corner_lower)): #self.mt_feats:#  np.arange(len(l1.corner_lower)):
            for dirn in ['left','right']:
                split_val=-99e9
                if self.split_weight!='univar_prob_distn' or (self.split_weight=='univar_prob_distn' and l1.size>0.000005): # don't split when it gets too small!!
                    if dirn=='right':
                        if l1.corner_lower[i_feat]<l2.corner_lower[i_feat] and l1.corner_upper[i_feat]>l2.corner_lower[i_feat] : # slice off bottom bit
                            split_val=l2.corner_lower[i_feat]
                    else: # left
                        if l1.corner_upper[i_feat]>l2.corner_upper[i_feat] and l1.corner_lower[i_feat]<l2.corner_upper[i_feat] :
                            split_val=l2.corner_upper[i_feat]
                if split_val==-99e9:
                    # do nothing, no need to partition on this feature and direction. Go to next feature/direction.
                    pass
                else: # need to split on this feat value
                    X=self.train_X[l1.train_data_idx]
                    y=self.train_y[l1.train_data_idx]
                    sample_weight=self.sample_weight[l1.train_data_idx]
                    [indx_left, indx_right,split_pt_act]=self.get_split_indxs(X,i_feat+1,split_val,self.feat_data_types[i_feat])
                    # calculate split weights
                    if self.split_weight=='parent_weight':
                        sample_wgt_left=l1.size# float(np.sum(sample_weight[indx_left]))
                        sample_wgt_right=l1.size#float(np.sum(sample_weight[indx_right]))
                    elif self.split_weight=='contained_pts_weight':
                        sample_wgt_left=np.max([0.5,float(np.sum(sample_weight[indx_left]))]) #l1.size/2.
                        sample_wgt_right=np.max([0.5,float(np.sum(sample_weight[indx_right]))]) #l1.size/2.
                    elif self.split_weight=='hybrid_prob' or self.split_weight=='hybrid_prob_empirical' or self.split_weight=='prob_empirical_cond' or self.split_weight=='hybrid_prob_empirical_orig_train' :
                        
                        sample_wgt_left=float(np.sum(sample_weight[indx_left]))
                        sample_wgt_right=float(np.sum(sample_weight[indx_right]))
                        if sample_wgt_left==0. or sample_wgt_right==0.:
                            if self.split_weight=='prob_empirical_cond':
                                [dist_vals,dist_probs]=self.univariate_distns[l1.predicted_class][i_feat]
                            else:
                                [dist_vals,dist_probs]=self.univariate_distns[i_feat]
                            left_extents=[l1.corner_lower[i_feat],split_val]
                            right_extents=[split_val,l1.corner_upper[i_feat]]
                            prob_left=calc_probability(dist_vals,dist_probs,left_extents[0],left_extents[1])
                            prob_right=calc_probability(dist_vals,dist_probs,right_extents[0],right_extents[1])
                            sample_wgt_left=l1.size*prob_left/(prob_left+prob_right)
                            sample_wgt_right=l1.size*prob_right/(prob_left+prob_right)

                    elif self.split_weight=='univar_prob_distn':
                        [dist_vals,dist_probs]=self.univariate_distns[i_feat]
                        left_extents=[l1.corner_lower[i_feat],split_val]
                        right_extents=[split_val,l1.corner_upper[i_feat]]
                        prob_left=calc_probability(dist_vals,dist_probs,left_extents[0],left_extents[1])
                        prob_right=calc_probability(dist_vals,dist_probs,right_extents[0],right_extents[1])
                        sample_wgt_left=l1.size*prob_left/(prob_left+prob_right)
                        sample_wgt_right=l1.size*prob_right/(prob_left+prob_right)
                        if l1.left.size==0:
                            print('what the 3')
                    
                    if self.split_criterion=='both_sides_have_pts':
                        split_decision=np.sum(sample_weight[indx_left])>0 and np.sum(sample_weight[indx_right])>0
                    elif self.split_criterion=='incomp_side_has_pts' : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                        if dirn=='right':
                            #B_TEST 2/2 
                            split_decision=np.sum(sample_weight[indx_left])>0 
                            #split_decision=len(indx_left)>0
                        else: # left
                            #B_TEST 2/2 
                            split_decision=np.sum(sample_weight[indx_right])>0 
                            #split_decision=len(indx_right)>0
                    elif self.split_criterion=='both_sides_have_min_sample_wgt' : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                        split_decision=sample_wgt_left>=self.min_split_weight and sample_wgt_right>=self.min_split_weight
                    else: #if self.split_criterion=='all_splits_ok' : 
                        split_decision=True
                    if split_decision:
                        change_made=True
                        l1.split()
                        l1.decision_feat=i_feat+1
                        l1.decision_data_type=self.feat_data_types[i_feat]
                        l1.decision_values=split_val

#                        if False: # NEW TECHNIQUE PRE 10/4/17 ####base technique - assign probs and tally of parent
#                            l1.left._probabilities=l1.probabilities.copy()
#                            l1.right._probabilities=l1.probabilities.copy()
#                            l1.left.tally=l1.tally.copy()
#                            l1.right.tally=l1.tally.copy()
#                        elif True: # NEW TECHNIQUE FROM 10/4/17
                            #l1.left._probabilities=l1.probabilities.copy()
                            #l1.right._probabilities=l1.probabilities.copy()
                        if self.split_class=='parent_class':
                            l1.left._probabilities=l1.probabilities.copy()
                            l1.right._probabilities=l1.probabilities.copy()
                            l1.left.tally=l1.tally.copy()
                            l1.right.tally=l1.tally.copy()
                        elif self.split_class=='contained_pts_class':
                            l1.left.update_ys(y[indx_left],sample_weight[indx_left])
                            l1.right.update_ys(y[indx_right],sample_weight[indx_right])
                            

                        #l1.left.update_ys(y[indx_left],sample_weight[indx_left])
                        #l1.right.update_ys(y[indx_right],sample_weight[indx_right])
                        l1.left.size=sample_wgt_left
                        l1.right.size=sample_wgt_right
                        l1.left.train_data_idx=l1.train_data_idx[indx_left]
                        l1.right.train_data_idx=l1.train_data_idx[indx_right]
                        #if l1.train_data_idx[indx_left][0]==31 or l1.train_data_idx[indx_right][0]==31:
                        #    print('problemo')
                        # old way to do it
                        #l1.left.size=l1.size/2.
                        #l1.right.size=l1.size/2.
                        #l1.left.train_data_idx=l1.train_data_idx
                        #l1.right.train_data_idx=l1.train_data_idx
                        
                        ## calculate upper and lower corners of new nodes
                        l1.left.corner_lower=l1.corner_lower.copy()
                        l1.left.corner_upper=l1.corner_upper.copy()
                        l1.right.corner_lower=l1.corner_lower.copy()
                        l1.right.corner_upper=l1.corner_upper.copy()
                        if l1.decision_data_type=='ordinal':
                            l1.left.corner_upper[i_feat]=split_val
                            l1.right.corner_lower[i_feat]=split_val
                        l1.left.resubst_err_node=l1.resubst_err_node #np.sum(self.sample_weight[indx_left])/self.sample_weight_total*(1-np.max(node.left.probabilities)) 
                        l1.right.resubst_err_node=l1.resubst_err_node #np.sum(self.sample_weight[indx_right])/self.sample_weight_total*(1-np.max(node.right.probabilities)) 
                        if np.isnan(l1.left._probabilities[0]) or np.isnan(l1.right._probabilities[0]):
                            print('sdfhwer')
                        # move to child to follow
                        l1=l1.left if dirn=='left' else l1.right

        return change_made
                    
    def grow_node(self,node,max_features,random_state,min_samples_split,min_samples_leaf,min_weight_leaf,max_depth,path=[],node_orig_num_nmt_pairs=0., require_abs_impurity_redn=True):
        if node is None:
            node=self.root_node
        # preliminary checks:
        this_ys=self.train_y[node.train_data_idx]
        #print ([min_samples_split,min_samples_leaf)
        if node.size<min_samples_split or node.size<min_samples_leaf*2 or node.depth>=max_depth or np.sum(this_ys==this_ys[0])==len(this_ys):
#            if np.max(node.probabilities)<1:
#                print('ouch')
            return 
        #

                   
        # add child nodes
        node.split()
        # get data at this node
        X=self.train_X[node.train_data_idx]
        y=self.train_y[node.train_data_idx]
        sample_weight=self.sample_weight[node.train_data_idx]
        # identify constant features to exclude from choice. Slightly different to scikit in that while scikit eliminates features known to be constant, it also allows discovers new features to be constant (depending on random draw) and these are counted towards max_features. The technique here eliminates all features known to be constant and hence may have slighly more leaves in the end.
        feats_non_const=[]
        CONSTANT_THRESHOLD=1e-7
        for ifeat in np.arange(self.n_features,dtype='int')+1:
            if (np.max(X[:,ifeat-1])-np.min(X[:,ifeat-1]))>CONSTANT_THRESHOLD:
               feats_non_const.append(ifeat) 
        # find the best feature to split this node on - like scikit, randomise feat order
        feats_to_test=random_state.permutation(feats_non_const) #feats_non_const if max_features==self.n_features else random_state.permutation(feats_non_const)[0:np.min([len(feats_non_const),max_features])]
        #feats_to_test=np.sort(feats_to_test)
        #print(feats_to_test)
        # include all feature including constant feats (like scikit learn)
#        feats_to_test=random_state.permutation(np.arange(self.n_features,dtype='int')+1)
#        CONSTANT_THRESHOLD=1e-7
        #print(feats_to_test)
        best=[-np.inf,None,None,None]
        #num_feats_tested=0
        feat_indx=0
        #at_least_one_non_const_split=False
        #for ifeat in feats_to_test:
        #best_split_feats=[]
        while feat_indx<len(feats_to_test) and feat_indx<max_features:# or not at_least_one_non_const_split ) :
            ifeat=feats_to_test[feat_indx]
            data_type=self.feat_data_types[ifeat-1]
            best_gain_this_feat=-np.inf
            #valid_split_exists=False
            #num_feats_tested=num_feats_tested+1
            #split_vals=np.sort(np.unique(X[:,ifeat-1]))
            # check if all feat vals are constant
            #if np.max(X[:,ifeat-1])-np.min(X[:,ifeat-1])<CONSTANT_THRESHOLD: # all feat vals constant
            #    pass
            #else: # all feat vals are NOT constant
            #at_least_one_non_const_split=True   
            if data_type=='ordinal':
                split_vals=self.split_pts[ifeat-1]#(split_vals[1:len(split_vals)]+split_vals[0:len(split_vals)-1])/2.0
                feat_val_lower=node.corner_lower[ifeat-1]
                feat_val_upper=node.corner_upper[ifeat-1]
                # S_TEST 2/3
                if feat_val_lower>-np.inf:   
                    split_vals=split_vals[split_vals>=feat_val_lower]
                if feat_val_upper<np.inf:  
                    split_vals=split_vals[split_vals<=feat_val_upper]
                indx_left_last=[-1]
                for split_pt in split_vals:
                    [indx_left, indx_right,split_pt_act]=self.get_split_indxs(X,ifeat,split_pt,data_type)
                    if len(indx_left)!=len(indx_left_last) or np.all(indx_left_last==-1):
                        # S_TEST 3/3 Find closest standard split point
                        split_pt=split_vals[min(range(len(split_vals)), key=lambda i: abs(split_vals[i]-split_pt_act))] 
                        if np.sum(sample_weight[indx_left])>=min_samples_leaf and np.sum(sample_weight[indx_right])>=min_samples_leaf  : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                            #valid_split_exists=True
                            node.left.update_ys(y[indx_left],sample_weight[indx_left])
                            node.right.update_ys(y[indx_right],sample_weight[indx_right])
                            # Entropy reduction
                            p=float(np.sum(sample_weight[indx_left]))/node.size#float(len(y[indx_left]))/node.size #p is the size of a child set relative to its parent
                            gain=node.impurity-p*node.left.impurity-(1-p)*node.right.impurity #cf. formula information gain 
                            if self.criterion=='info_gain_ratio':
                                gain=gain/intrinsic_value([p,1.-p])
                            
                            if gain>best[0]: 
                                best=[gain,ifeat,split_pt,0] 
                            if gain>best_gain_this_feat:
                                best_gain_this_feat=gain
                        indx_left_last=indx_left
                                
            elif data_type=='nominal':
                values=self.feat_vals[ifeat-1] #np.sort(np.unique(X[:,ifeat-1]))
                #valid_split_exists=False
                for subset_size in np.arange(1,len(values)):
                    for val_subset in combinations(values, subset_size):
                        if np.sum(sample_weight[indx_left])>=min_samples_leaf and np.sum(sample_weight[indx_right])>=min_samples_leaf  : #len(indx_left)>0 and len(indx_right)>0  : #CHANGE FROM SOLVED VERSION 10/4/17: (len(indx_left)>0 and dirn=='right') or  (len(indx_right)>0 and dirn=='left') :
                            #valid_split_exists=True
                            node.left.update_ys(y[indx_left])
                            node.right.update_ys(y[indx_right])
                            # Information gain
                            p=float(np.sum(sample_weight[indx_left]))/node.size #p is the size of a child set relative to its parent
                            gain=node.impurity-p*node.left.impurity-(1-p)*node.right.impurity #cf. formula information gain
                            if gain>best[0]: 
                                best=[gain,ifeat,val_subset,[item for item in values if item not in val_subset]  ] 
                            if gain>best_gain_this_feat:
                                best_gain_this_feat=gain
            #if valid_split_exists: # count all non-constant splits #valid_split_exists and best_gain_this_feat>=0 :
            #num_feats_tested=num_feats_tested+1
            feat_indx=feat_indx+1
        #print('num feats checked: ' + str(feat_indx))
        if require_abs_impurity_redn:
            stop=best[0]<0
            #if best[0]<=0 and best[0]!=-np.inf:
            #    print('terminated due to req abs gain' + str(best[0]))
        else:
            stop=best[0]==-np.inf
            #print('terminated due to no valid partitions')
        #print('best split feats: ' + str(best_split_feats))
        if stop: # ANY CHANGE IS OK - NO LONGER REQUIRES A GAIN IN INFORMATION AS DESCRIBED BY   #best[0]<=0: # no improvement made by split
            node.fuse()
            return
        else: # update children to final
            node.decision_feat=best[1]
            node.decision_data_type=self.feat_data_types[node.decision_feat-1]
            node.decision_values=best[2]
            [indx_left, indx_right,split_pt_act]=self.get_split_indxs(X,node.decision_feat,node.decision_values,node.decision_data_type)
            node.left.update_ys(y[indx_left],sample_weight[indx_left])
            node.right.update_ys(y[indx_right],sample_weight[indx_right])
            node.left.train_data_idx=node.train_data_idx[indx_left]
            node.right.train_data_idx=node.train_data_idx[indx_right]
            # calculate upper and lower corners of new nodes
            node.left.corner_lower=node.corner_lower.copy()
            node.left.corner_upper=node.corner_upper.copy()
            node.right.corner_lower=node.corner_lower.copy()
            node.right.corner_upper=node.corner_upper.copy()
            node.left.resubst_err_node=np.sum(self.sample_weight[indx_left])/self.sample_weight_total*(1-np.max(node.left.probabilities)) 
            node.right.resubst_err_node=np.sum(self.sample_weight[indx_right])/self.sample_weight_total*(1-np.max(node.right.probabilities)) 
            if node.decision_data_type=='ordinal':
                node.left.corner_upper[node.decision_feat-1]=node.decision_values
                node.right.corner_lower[node.decision_feat-1]=node.decision_values
            else: # nominal
                node.left.corner_upper[node.decision_feat-1]=node.decision_values#best[3] # inverse set
                node.left.corner_lower[node.decision_feat-1]=node.decision_values
                node.right.corner_upper[node.decision_feat-1]=node.decision_values#best[3] # inverse set
                node.right.corner_lower[node.decision_feat-1]=node.decision_values
            # update tree properties
            self.impurity=self.impurity-node.impurity*node.size/self.size +node.left.impurity*node.left.size/self.size+node.right.impurity*node.right.size/self.size
            self.grow_node(node.left,max_features,random_state,min_samples_split,min_samples_leaf,min_weight_leaf,max_depth,require_abs_impurity_redn=require_abs_impurity_redn)
            self.grow_node(node.right,max_features,random_state,min_samples_split,min_samples_leaf,min_weight_leaf,max_depth,require_abs_impurity_redn=require_abs_impurity_redn)
            return
    def manually_calc_num_leaves(self,node=None):
        if node is None: node=self.root_node
        if node.is_leaf():
            return 1
        else:
            left_ttl=self.manually_calc_num_leaves(node.left) 
            right_ttl=self.manually_calc_num_leaves(node.right) 
            return left_ttl + right_ttl  
    def manually_get_leaf_nodes(self,node=None):
        if node is None: node=self.root_node
        if node.is_leaf():
            return [node]
        else:
            return self.manually_get_leaf_nodes(node.left) +self.manually_get_leaf_nodes(node.right)
            
    def calc_num_nmt_pairs(self,node,starting_from_node=None):
        if starting_from_node is None: starting_from_node=self.root_node 
        if starting_from_node.is_leaf():
            return 2. if self.is_non_monotone(node,starting_from_node) else 0.
        else:
            left_ttl=self.calc_num_nmt_pairs(node,starting_from_node.left) 
            right_ttl=self.calc_num_nmt_pairs(node,starting_from_node.right) 
            return left_ttl + right_ttl
            
    def is_non_monotone(self,node1,node2):
        if self.check_increasing_bool(node1.corner_lower,node1.corner_upper,node2.corner_lower,node2.corner_upper,self.in_feats,self.de_feats,self.nmt_feats):
            return np.int(node1.predicted_class>node2.predicted_class)
        elif self.check_increasing_bool(node2.corner_lower,node2.corner_upper,node1.corner_lower,node1.corner_upper,self.in_feats,self.de_feats,self.nmt_feats):
            return np.int(node2.predicted_class>node1.predicted_class)
        else:
            return 0.
    def get_split_indxs(self,X,ifeat,split_values,data_type):
        if data_type=='ordinal':
            if len(X.shape)==2:
                indx_left=np.arange(X.shape[0])[X[:,ifeat-1]<=split_values]
                indx_right=np.arange(X.shape[0])[X[:,ifeat-1]>split_values]
                if len(indx_left)>0 and len(indx_right)>0:
                    split_pt_act=(np.max(X[indx_left,ifeat-1])+np.min(X[indx_right,ifeat-1]))/2.
                else:
                    split_pt_act=split_values
            else:
                indx_left=np.arange(1) if X[ifeat-1]<=split_values else np.arange(0) 
                indx_right=np.arange(1) if X[ifeat-1]>split_values else np.arange(0)
                split_pt_act=split_values #(np.max(X[indx_left,ifeat-1])+np.min((X[indx_right,ifeat-1]))/2.
        else:
            if len(X.shape)==2:
                indx_bool_left=np.asarray([( X[i,ifeat-1] in split_values ) for i in np.arange(X.shape[0])],dtype='bool')
                indx_bool_right= indx_bool_left==False
                indx_left=np.arange(X.shape[0])[indx_bool_left]
                indx_right=np.arange(X.shape[0])[indx_bool_right]
            else: # treat as one row
                indx_left=np.arange(1) if X[ifeat-1] in split_values else np.arange(0)# X.shape[0])[[(X[i,ifeat-1] in split_values) for i in np.arange(X.shape[0])]]
                indx_right=np.arange(1) if X[ifeat-1] not in split_values else np.arange(0)
            split_pt_act=split_values
        return [indx_left,indx_right,split_pt_act]
            
    def printtree(self,node=None,indent=''):
        if node is None:
            node=self.root_node
        # Is this a leaf node?
        if node.is_leaf():
            print('Leaf ' + str(node.index_leaf) +': ' + str(node._probabilities)  + str(node.tally))# + ' ' + str(node.corner_lower) + ' ' + str(node.corner_upper))
        else:
            print('feat_' + str(node.decision_feat)+''+('<=' if node.decision_data_type=='ordinal' else ' in ')+str(node.decision_values)+'? ' + str(node.tally))
            # Print the branches
            print(indent+'T->', end=" ")
            self.printtree(node.left,indent+'  ')
            print(indent+'F->', end=" ")
            self.printtree(node.right,indent+'  ')
    
    @property
    def feat_vals(self):
        if self._feat_vals is None:
            if not self.train_X is None: # attempt to extract feat data types
                fv=[]
                for ifeat in np.arange(self.train_X.shape[1])+1:
                    feat_vals=np.sort(np.unique(self.train_X[:,ifeat-1]))
                    if np.sum(np.abs(np.asarray(feat_vals,dtype='int')-feat_vals))<1e-9:
                        feat_vals=np.asarray(feat_vals,dtype='int')
                    fv.append(tuple( feat_vals))
                self._feat_vals=fv
        return self._feat_vals    
    @property
    def feat_data_types(self):
        if self._feat_data_types=='auto':
            if not self.train_X is None: # attempt to extract feat data types
                feat_data_types=[]
                for ifeat in np.arange(self.train_X.shape[1]): # all treated as ordinal
                    feat_data_types.append('ordinal')
#                    feat_vals=np.unique(np.ravel(self.train_X[:,ifeat]))
#                    if False and np.sum(np.abs(np.asarray(feat_vals,dtype='int')-feat_vals))<1e-9: # ELIMINATE NOMINAL FEAT DETECTION FOR NOW
#                        #print(str(feat_vals) +str(np.sum(np.abs(np.arange(np.min(feat_vals),np.min(feat_vals)+len(feat_vals),dtype='int')-feat_vals))<1e-9))
#                        if np.sum(np.abs(np.arange(np.min(feat_vals),np.min(feat_vals)+len(feat_vals),dtype='int')-feat_vals))<1e-9:
#                            if len(feat_vals)<=2:
#                                feat_data_types[ifeat]='nominal'
                self._feat_data_types=feat_data_types
        return self._feat_data_types
        
    @property
    def feat_labels(self):
        if self._feat_labels=='auto':
            self._feat_labels=[str(i) for i in np.arange(self.n_features)+1 ]
        return self._feat_labels            
        
    @timeit # about 30% faster than the simple technique
    def get_increasing_leaf_node_pairs_new_old   (self):
        probs,lowers,uppers=self.get_corner_matrices()
        G=nx.DiGraph()
        in_feats=np.asarray(self.in_feats)
        de_feats=np.asarray(self.de_feats)
        mt_feats=list(in_feats).copy()
        for i in np.arange(len(de_feats)): mt_feats.append(de_feats[i])
        nmt_feats=[f for f in np.arange(self.n_features) if f not in mt_feats]
        # initialise graph with comparison of leaf 0 with all other leaves:
        n_leaves=len(self.leaf_nodes)
        for i in np.arange(n_leaves):
            # get master list of comparable leaves
            smart_filter=True # 10% faster... but it is faster
            if smart_filter:
                last_comparible_node=self.drop_down_hypercube(lowers[i,:], uppers[i,:])
                leaves_to_check=set(self.get_leaf_ids_under(last_comparible_node))
            else:
                leaves_to_check=set(np.arange(n_leaves))
            leaves_to_check.remove(i)
            # check outgoing
            check=leaves_to_check.copy()
            if i in G.nodes():  
                check.difference_update(nx.ancestors(G,i))
                check.difference_update(nx.descendants(G,i))
            while len(check)>0:
                j=check.pop()
                if self.check_increasing_bool(lowers[i,:], uppers[i,:],lowers[j,:], uppers[j,:],in_feats,de_feats,nmt_feats):
                    #G.add_node(j)
                    G.add_edge(i,j)
                    check.difference_update(nx.descendants(G,j))
            # check incoming
            check=leaves_to_check.copy()
#            check=set(np.arange(n_leaves))
#            check.remove(i)
            if i in G.nodes(): 
                check.difference_update(nx.ancestors(G,i))
                check.difference_update(nx.descendants(G,i))
            while len(check)>0:
                j=check.pop()
                if self.check_increasing_bool(lowers[j,:], uppers[j,:],lowers[i,:], uppers[i,:],in_feats,de_feats,nmt_feats):#(,uppers[i,:]):
                    #G.add_node(j)
                    G.add_edge(j,i)
                    check.difference_update(nx.ancestors(G,j))
        return G.edges()
         
    def get_increasing_leaf_node_pairs_new(self):
        probs,lowers,uppers=self.get_corner_matrices()
    
        max_pairs=int(np.round(lowers.shape[0]*lowers.shape[0]))
        incr_pairs=np.zeros([max_pairs,2],dtype=np.int32)
        n_pairs_new=isoensemble.get_increasing_leaf_node_pairs(lowers,uppers,self.mt_feat_types,incr_pairs)
        #incr_pairs_old=self.get_increasing_leaf_node_pairs_simple()
        return incr_pairs[0:n_pairs_new,:]
    @timeit
    def get_increasing_leaf_node_pairs_simple  (self):
        probs,lowers,uppers=self.get_corner_matrices()
        G=nx.DiGraph()
        in_feats=self.in_feats
        de_feats=self.de_feats #np.asarray(self.decr_feats)-1
        mt_feats=list(in_feats).copy()
        for i in np.arange(len(de_feats)): mt_feats.append(de_feats[i])
        nmt_feats=[f for f in np.arange(self.n_features) if f not in mt_feats]
        # initialise graph with comparison of leaf 0 with all other leaves:
        n_leaves=len(self.leaf_nodes)
        for i in np.arange(n_leaves):
            for j in np.arange(n_leaves):
                if i!=j:
                    if self.check_increasing_bool(lowers[i,:], uppers[i,:],lowers[j,:], uppers[j,:],in_feats,de_feats,nmt_feats):
                        G.add_edge(i,j)
                    if self.check_increasing_bool(lowers[j,:], uppers[j,:],lowers[i,:], uppers[i,:],in_feats,de_feats,nmt_feats):#(,uppers[i,:]):
                        G.add_edge(j,i) 
        return G.edges()
        
#    def get_increasing_leaf_node_pairs  (self):
#        pairs=[]
#        for ileaf in np.arange(len(self.leaf_nodes)):
#            leaf1=self.leaf_nodes[ileaf]
#            for jleaf in np.arange(ileaf+1,len(self.leaf_nodes)):
#                if ileaf !=jleaf:
#                    leaf2=self.leaf_nodes[jleaf]
#                    if self.check_if_increasing(leaf1, leaf2): # if true leaf1<leaf2
#                         pairs.append([leaf1.index_leaf,leaf2.index_leaf])       
#                    elif self.check_if_increasing(leaf2, leaf1): # if true leaf1<leaf2
#                         pairs.append([leaf2.index_leaf,leaf1.index_leaf])  
#        return pairs
    
    def check_increasing_bool(self,n1_lower, n1_upper,n2_lower, n2_upper,mt_incr_feats,mt_decr_feats,nmt_feats):
        if len(mt_incr_feats)>0:
            mt_incr=np.all(np.asarray(n1_lower)[mt_incr_feats]<np.asarray(n2_upper)[mt_incr_feats])
        else:
            mt_incr=True
        if mt_incr:
            if len(mt_decr_feats)>0:
                mt_decr=np.all(np.asarray(n1_upper)[mt_decr_feats]>np.asarray(n2_lower)[mt_decr_feats])
            else:
                mt_decr=True
            if mt_decr:
                if len(nmt_feats)==0:
                    return True
                elif np.all(np.asarray(n1_upper)[nmt_feats]>np.asarray(n2_lower)[nmt_feats]):
                    return np.all(np.asarray(n2_upper)[nmt_feats]>np.asarray(n1_lower)[nmt_feats])
                else:
                    return False
            else:
                return False
        else:
            return False
                
#    def check_if_increasing(self,node1, node2):
#        mt_feats=self.incr_feats.copy()
#        for i in np.arange(len(self.decr_feats)): mt_feats.append(self.decr_feats[i])
#        nmt_feats=[f for f in np.arange(self.n_features)+1 if f not in mt_feats]
#        if len(mt_feats)==0:
#            return False
#        else:
#            n1_lower=np.asarray(node1.corner_lower.copy(),dtype='float')
#            n2_upper=np.asarray(node2.corner_upper.copy(),dtype='float')
#            n2_lower=np.asarray(node2.corner_lower.copy(),dtype='float')
#            n1_upper=np.asarray(node1.corner_upper.copy(),dtype='float')
#            # test incr feats:
#            if len(self.incr_feats)>0:
#                mt_increasing_feats=np.sum(n1_lower[np.asarray(self.incr_feats)-1]<n2_upper[np.asarray(self.incr_feats)-1])==len(self.incr_feats)
#                if not mt_increasing_feats:
#                    return False
#            # test decr feats:
#            if len(self.decr_feats)>0:
#                mt_decreasing_feats=np.sum(n1_upper[np.asarray(self.decr_feats)-1]>n2_lower[np.asarray(self.decr_feats)-1])==len(self.decr_feats)
#                if not mt_decreasing_feats:
#                    return False
#            # check partial monotonicity overap in unconstrained features
#            if len(nmt_feats)==0:
#                nmt_overlap=True
#            else:
#                nmt_overlap=np.sum(n1_upper[np.asarray(nmt_feats)-1]>n2_lower[np.asarray(nmt_feats)-1])==len(nmt_feats)
#                nmt_overlap=nmt_overlap and np.sum(n1_lower[np.asarray(nmt_feats)-1]<n2_upper[np.asarray(nmt_feats)-1])==len(nmt_feats)
#            return nmt_overlap 
    def get_corner_matrices(self):
        n_leaves=len(self.leaf_nodes)
        lower=np.zeros([n_leaves, self.n_features],dtype=np.float64)
        upper=np.zeros([n_leaves, self.n_features],dtype=np.float64)
        probabilities=np.zeros([n_leaves, self.n_classes])
        for i in np.arange(n_leaves):
            lower[i,:]=self.leaf_nodes[i].corner_lower.copy()
            upper[i,:]=self.leaf_nodes[i].corner_upper.copy()
            if len(self.leaf_nodes[i].probabilities)==0:
                print('guh!!')
            probabilities[i,:]=self.leaf_nodes[i].probabilities.copy()
        return [probabilities,lower,upper]
        
    def drop_down_hypercube(self,lower, upper,node=None):
        if node is None: node=self.root_node
        if node.is_leaf():
            return node
        elif upper[node.decision_feat-1]<=node.decision_values:
            return self.drop_down_hypercube(lower, upper,node.left)
        elif lower[node.decision_feat-1]>node.decision_values:
            return self.drop_down_hypercube(lower, upper,node.right)   
        else:
            return node           
          
    def get_leaf_ids_under(self,node):#,leaf_ids=[]):
        if node.index==self.root_node.index:
            return np.arange(len(self.leaf_nodes))
        elif node.is_leaf():
            return np.asarray([node.index_leaf])
        else:
            leaves=[]
            for leaf in self.leaf_nodes:
                if node.index in leaf.path:
                    leaves.append(leaf.index_leaf)
            return np.asarray(leaves)
#        if node.is_leaf():
#            #leaf_ids.append(node.index_leaf)
#            return [node.index_leaf] #leaf_ids.copy()
#        else:
#            leaf_ids_left= self.get_leaf_ids_under(node.left,leaf_ids).copy()
#            leaf_ids_right=self.get_leaf_ids_under(node.right,leaf_ids).copy()
#            return leaf_ids_left+leaf_ids_right
    def number_nodes(self,calc_resubs_err=False):
        if not self.root_node is None:
            self.leaf_nodes=[]
            self.branch_nodes=[]
            queue=deque()
            queue.append(self.root_node)
            index=0
            index_leaf=0
            while len(queue)>0:
                node=queue.popleft()
                node.resubst_err_branch=0
                node.num_leaves_branch=0
                node.index=index
                if node.index!=self.root_node.index:
                    node.path=node.parent.path+[node.parent.index]
                index=index+1
                if node.is_leaf():
                    node.index_leaf=index_leaf
                    self.leaf_nodes.append(node)
                    index_leaf=index_leaf+1
                    if calc_resubs_err:
                        if node.parent!=None: # not root node
                            self.propagate_leaf_data_to_parents(node,node.parent)
                else:
                    self.branch_nodes.append(node)
                    queue.append(node.left)
                    queue.append(node.right)
            if len(self.leaf_nodes)>self.peak_leaves:
                self.peak_leaves=len(self.leaf_nodes)
                            
    
    def propagate_leaf_data_to_parents(self,node,parent):
        parent.resubst_err_branch=parent.resubst_err_branch+node.resubst_err_node
        parent.num_leaves_branch=parent.num_leaves_branch+1
        if parent.parent!=None:# notroot node:
            self.propagate_leaf_data_to_parents(node,parent.parent)
        
    def drop_down(self,node,Xi,path):
        path.append(node.index)
        if node.is_leaf():
            probs=[]
            probs_raw=node.probabilities
            i_raw_prob=0
            for i in np.arange(len(self.classes)):
                if i in node.classes: #tally.keys():
                    probs.append(probs_raw[i_raw_prob])
                    i_raw_prob=i_raw_prob+1
                else:
                    probs.append(0.)
            return [node.index_leaf,probs,path]
        else: # not a leaf node, evaluate and send on way
            [indx_left, indx_right,split_pt_act]=self.get_split_indxs(Xi,node.decision_feat,node.decision_values,node.decision_data_type)
            return self.drop_down(node.left if indx_left==[0] else node.right, Xi,path)
        return            
    def apply_base(self,X):
        probs=np.zeros([X.shape[0],len(self.classes)]) 
        paths=[]
        ileaf_indexes=np.zeros(X.shape[0]) 
        for i in np.arange(X.shape[0]):
            if i==26:
                pass
            [ileaf,probs_,path]=self.drop_down(self.root_node,X[i,:],[])
            probs[i,:]=probs_
            paths.append(path)
            ileaf_indexes[i]=ileaf
        return [ileaf_indexes,probs,paths]
        
    def predict_proba(self,X):
        [ileaf,probs_,paths]=self.apply_base(X)
        return probs_
        
    def apply(self,X):
        [ileaf,probs_,paths]=self.apply_base(X)
        return ileaf
        
    def get_pairs_to_split(self,pm_pairs_clean,nmt_pairs,normalise_nmt_nodes):
        if normalise_nmt_nodes==0:
            pairs_to_split=[] #if normalise_nmt_nodes==1 else pm_pairs_clean
        elif normalise_nmt_nodes==1:
            
            pairs_to_split=nmt_pairs 
        elif normalise_nmt_nodes==2:
            pairs_to_split=pm_pairs_clean
        elif normalise_nmt_nodes==3:
            pairs_to_split=self.get_non_monotone_pairs_extended(nmt_pairs,pm_pairs_clean)
        return pairs_to_split
#    def set_leaf_sizes(self,sizes):
#        i=0
#        for leaf in self.leaf_nodes:
#            leaf.size=sizes[0]
#            i=i+1
    # returns 0: no changes required to monotonise
    # 1: changes made to monotonise
    @timeit
    def monotonise(self,incr_feats,decr_feats,sample_reweights=None,normalise_nmt_nodes=0,split_criterion=None, split_class=None,split_weight=None,min_split_weight=0.5,univariate_distns=None):
        self.set_mt_feats(incr_feats,decr_feats)
        self.split_criterion=split_criterion
        self.split_class=split_class
        self.split_weight=split_weight
        self.min_split_weight=min_split_weight
        self.univariate_distns=univariate_distns
        #self.incr_feats=incr_feats.copy()
        #self.decr_feats=decr_feats.copy()
        use_latest_nmt_pairs=True
        if use_latest_nmt_pairs:
            # WAY FOR MOST SOLVES:
            pm_pairs=self.get_increasing_leaf_node_pairs_new()   
            pm_pairs_clean=self.eliminate_unnecessary_incr_pairs(pm_pairs)
            nmt_pairs=self.get_non_monotone_pairs(pm_pairs_clean)
        else:
            # ALTERNATE SIMPLE WAY
            pm_pairs=self.get_increasing_leaf_node_pairs_simple()
            pm_pairs_clean=pm_pairs
            nmt_pairs=self.get_non_monotone_pairs(pm_pairs)
        
        # pairs check:
#        pm_pairs_simple=self.get_increasing_leaf_node_pairs_simple()   
#        pm_pairs_clean_simple=self.eliminate_unnecessary_incr_pairs(pm_pairs_simple)
#        print([len(pm_pairs_clean_simple),len(pm_pairs_clean)])
        
        #print(len(pm_pairs_clean))

        #nmt_pairs_extended=self.get_non_monotone_pairs_extended(nmt_pairs,pm_pairs_clean)
        pairs_to_split=self.get_pairs_to_split(pm_pairs_clean,nmt_pairs,normalise_nmt_nodes) #nmt_pairs if normalise_nmt_nodes==1 else pm_pairs_clean

                    
        #print('Num pairs [orig,clean,nmt,super_clean]: ' + str([len(pm_pairs),len(pm_pairs_clean),len(nmt_pairs),len(cleaned_pairs)]))
        if nmt_pairs==[]: # already monotone
            return 0 
        if normalise_nmt_nodes>0:
            if False:
                ###### TECHNIQUE A - very slow due to continually recalculating get_increasing_leaf_node_pairs_new() ##########
                # pick first nmt edge and normalise nodes
                keep_going=True
                while keep_going:
                    keep_going=False
                    for pair in nmt_pairs:
                        change_made1=self.grow_segregated_nodes(self.leaf_nodes[pair[0]],self.leaf_nodes[pair[1]])
                        change_made2=self.grow_segregated_nodes(self.leaf_nodes[pair[1]],self.leaf_nodes[pair[0]])
                        if change_made1 or change_made2:
                            self.number_nodes()
                            #print(' now: ' + str(len(self.leaf_nodes)))
                            if use_latest_nmt_pairs:
                                # WAY FOR MOST SOLVES:
                                pm_pairs=self.get_increasing_leaf_node_pairs_new()   
                                pm_pairs_clean=self.eliminate_unnecessary_incr_pairs(pm_pairs)
                                nmt_pairs=self.get_non_monotone_pairs(pm_pairs_clean)
                            else:
                                # ALTERNATE SIMPLE WAY
                                pm_pairs=self.get_increasing_leaf_node_pairs_simple()
                                pm_pairs_clean=pm_pairs
                                nmt_pairs=self.get_non_monotone_pairs(pm_pairs)
                            keep_going=True
                            break
            
            else:
                ####### TECHNIQUE B - between 50% and 85% faster ##########                 
                keep_going=True
                while keep_going:
                    keep_going=False
                    change_made=False
                    changed_nodes=[]
                    #pairs_to_split=nmt_pairs if normalise_nmt_nodes==1 else pm_pairs_clean
                    for pair in pairs_to_split:
                        if True:# pair[0] not in changed_nodes and pair[1] not in changed_nodes : # this filter seems a tad slower and hence complexity is not warranted
                            change_made1=self.grow_segregated_nodes(self.leaf_nodes[pair[0]],self.leaf_nodes[pair[1]])
                            change_made2=self.grow_segregated_nodes(self.leaf_nodes[pair[1]],self.leaf_nodes[pair[0]])
                            if change_made1: changed_nodes.append(pair[0])
                            if change_made2: changed_nodes.append(pair[1])
                            change_made=change_made or change_made1 or change_made2
    
                    if change_made:
                        self.number_nodes()
                        #print(' now: ' + str(len(self.leaf_nodes)))
                        pm_pairs=self.get_increasing_leaf_node_pairs_new()   
                        pm_pairs_clean=self.eliminate_unnecessary_incr_pairs(pm_pairs)
                        nmt_pairs=self.get_non_monotone_pairs(pm_pairs_clean)
                        
                        pairs_to_split=self.get_pairs_to_split(pm_pairs_clean,nmt_pairs,normalise_nmt_nodes) #nmt_pairs if normalise_nmt_nodes==1 else pm_pairs_clean
                        keep_going=True
                    #break  
                        
                        
                    #nmt_graph=nx.DiGraph()
                    #nmt_graph.add_edges_from(nmt_pairs)
#                    for nd in nmt_graph.nodes():
#                        if len(nmt_graph.neighbors(nd))==1:
#                            change_made=self.grow_segregated_nodes(self.leaf_nodes[nd],self.leaf_nodes[ nmt_graph.neighbors(nd)[0] ])
            #print(' Final leaf nodes: ' + str(len(self.leaf_nodes)))                
        #else: # we have non-monotone pairs
        cleaned_pairs=self.clean_monotone_island_pairs(pm_pairs_clean,nmt_pairs)
        if sample_reweights is None:
            weights=self.get_leaf_sizes()
        else:
            weights=self.recalc_leaf_sizes(sample_reweights)

        cdf=self.get_cum_probabilities()
        # solve new pdfs
        cdf_iso=np.ones(cdf.shape) 
        pdf_iso=np.zeros(cdf.shape) 
        cum_sse=0.          
        for i_class in np.arange(cdf.shape[1]):
            probs_class=cdf[:,i_class]
            gir=isoensemble.GeneralisedIsotonicRegression()
            if i_class<cdf.shape[1]-1:
                #cdf_iso[:,i_class]=gir.fit(probs_class,pm_pairs_clean,sample_weight=weights,increasing=False)
                #print(probs_class)
                cdf_iso[:,i_class]=np.round(gir.fit(probs_class,cleaned_pairs,sample_weight=weights,increasing=False),6)
            if i_class==0:
                pdf_iso[:,i_class]=cdf_iso[:,i_class]
            else:
                pdf_iso[:,i_class]=cdf_iso[:,i_class]-cdf_iso[:,i_class-1]
        cum_sse=np.sum((cdf_iso-cdf)**2)
        # update leaf probabilities
        if cum_sse>1e-7: # some changes were made
            for leaf in self.leaf_nodes:
                leaf._probabilities=list(pdf_iso[leaf.index_leaf,:])
                if np.isnan(leaf._probabilities[0]):
                    print('what the')
            res= 1
        else: # effectively no changes made
            res= 0
        #print(cdf-cdf_iso)
        # check we now have monotone tree:
        nmt_pairs2=self.get_non_monotone_pairs(pm_pairs_clean)
        if len(nmt_pairs2)>0:
            print('ERROR: orig nmt pairs:' + str(len(nmt_pairs)) + " now: " + str(len(nmt_pairs2)))
        return res
    
    def get_leaf_sizes(self):
        weights=np.zeros(len(self.leaf_nodes))
        for leaf in self.leaf_nodes:
            weights[leaf.index_leaf]=leaf.size
        return weights
        
    def recalc_leaf_sizes(self,new_orig_sample_weights):
        weights=np.zeros(len(self.leaf_nodes))
        for leaf in self.leaf_nodes:
            for i in leaf.train_data_idx:
                weights[leaf.index_leaf]=weights[leaf.index_leaf] +new_orig_sample_weights[i] #leaf.size
        return weights
        
    def get_cum_probabilities(self)    :
        cdf=np.zeros([len(self.leaf_nodes),len(self.classes)])
        for leaf in self.leaf_nodes:
            for i_class in np.arange(len(self.classes)):
                if i_class==0:
                    cdf[leaf.index_leaf,i_class]=leaf.probabilities[0]                    
                else:
                    cdf[leaf.index_leaf,i_class]=cdf[leaf.index_leaf,i_class-1]+leaf.probabilities[i_class]  
        return cdf
    def extend_graph_until(self,master_graph,output_graph,pair,dirn,terminate_at):
        if dirn==1:
            next_nodes=master_graph.successors(pair[1])
            for inode in next_nodes:
                if self.leaf_nodes[inode].predicted_class!=terminate_at:
                    new_pair=[pair[1],inode]
                    output_graph.add_edge(new_pair[0],new_pair[1])
                    self.extend_graph_until(master_graph,output_graph,new_pair,dirn,terminate_at)
        else:
            next_nodes=master_graph.predecessors(pair[0])
            for inode in next_nodes:
                if self.leaf_nodes[inode].predicted_class!=terminate_at:
                    new_pair=[inode,pair[0]]
                    output_graph.add_edge(new_pair[0],new_pair[1])
                    self.extend_graph_until(master_graph,output_graph,new_pair,dirn,terminate_at)
        return
    @timeit
    def clean_monotone_island_pairs(self,pm_pairs_clean,nmt_pairs):
        graph=nx.DiGraph()
        graph.add_edges_from(pm_pairs_clean) 
        ud_graph=graph.to_undirected()
        nodes_with_constraints =set(graph.nodes())
        unchecked_nodes=nodes_with_constraints.copy()
        polluted_nodes=set(np.unique(np.ravel(np.asarray(nmt_pairs))))
        safe_island_nodes_to_remove=[]#set()
        for n in  graph.nodes():
            if graph.predecessors(n) == []: # root node #successors(n)
                if n in unchecked_nodes:
                    nodes=set(nx.descendants(ud_graph,n))
                    has_no_nmt_polluted_nodes = len(nodes.intersection(polluted_nodes))==0
                    if has_no_nmt_polluted_nodes:
                        safe_island_nodes_to_remove=safe_island_nodes_to_remove+ list(nodes) + [n]
                    unchecked_nodes.difference_update(nodes)
                    unchecked_nodes.difference_update([n])
        cleaned_edges=nx.DiGraph()
        for edge in pm_pairs_clean:
            if edge[0] not in  safe_island_nodes_to_remove :
                cleaned_edges.add_edge(edge[0],edge[1])                         
        return cleaned_edges.edges()
        
    @timeit
    def get_non_monotone_pairs_extended(self,nmt_pairs,pairs):
        # extends forward and backward in from given directly nmt_pairs to region that needs to be constrained
        graph=nx.DiGraph()
        graph.add_edges_from(nmt_pairs) 
        master_graph=nx.DiGraph()
        master_graph.add_edges_from(pairs) 
        for nmt_pair in nmt_pairs:
            self.extend_graph_until(master_graph,graph,nmt_pair,dirn=+1,terminate_at=self.classes[1])
            self.extend_graph_until(master_graph,graph,nmt_pair,dirn=-1,terminate_at=self.classes[0])
        # remove redundant edges
        result=self.eliminate_unnecessary_incr_pairs(graph.edges().copy())
        return result
    @timeit    
    def get_non_monotone_pairs(self,pm_pairs):
        nmt_pairs=[]
        for pair in pm_pairs:
            if self.leaf_nodes[pair[0]].predicted_class>self.leaf_nodes[pair[1]].predicted_class:
                nmt_pairs.append(pair)
        return nmt_pairs
    @timeit    
    def eliminate_unnecessary_incr_pairs(self,pm_pairs):
        G=nx.DiGraph()
        G.add_nodes_from(np.arange(len(self.leaf_nodes),dtype='int'))
        G.add_edges_from(pm_pairs)
        #nx.draw(G,with_labels=True)
        remove_redundant_edges(G)
        #plt.figure()
        #nx.draw(G,with_labels=True)
        return G.edges().copy()
    # find any pairse of child nodes with a common parent and the same predicted class. If found, fuse the parent.
    # returns 0 if no nodes were fused, or else the number of nodes fused.
    def simplify(self,node=None):
        if node is None: node=self.root_node
        if node.is_leaf():
            return 0
        elif node.left.is_leaf() and node.right.is_leaf(): # this is an immediate parent node
            if node.left.predicted_class==node.right.predicted_class:
                if node.predicted_class!=node.left.predicted_class: # these were obviously changed. set probabilities of node to average of children
                    node._probabilities=list((np.asarray(node.left.probabilities)*node.left.size+np.asarray(node.right.probabilities)*node.right.size)/(node.left.size+node.right.size))
                    if np.isnan(node._probabilities[0]):
                        print('what the')
                node.fuse()
                self.number_nodes()
                return 1
            else:
                return 0
        else: # node has children with different
            res_left=self.simplify(node.left)
            res_right=self.simplify(node.right)
            return res_left+res_right
class IsoDecisionTreeClassifier(BaseEstimator):
    """PMDecisionTreeClassifier ie a pure python decision tree with partial monotonicity capacity.
    LIMITATIONS: 
    1. Sample Weights are assumed to be integers corresponding to number of samples. Using other sample weights will result in unpredicctable behaivour.
    OPPORTUNITIES:
    1. Improve split efficiency by sorting datapoints
    """

    def __init__(self,
                 criterion='gini_l1', # note 'info_gain_ratio' doesn't work with order_ambiguity_weight_R>0 (entropy will be used, because gain ratio not scaled to matched the order-ambiguity score)
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0,
                 max_features=None,
                 max_leaf_nodes=None,
                 random_state=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 presort=False,
                 feat_data_types='auto',
                 incr_feats=None,
                 decr_feats=None,
                 monotonicity_type='ict', #'ict' or 'order_ambiguity'
                 monotonicity_params=None,
                 split_pts=None,
                 normalise_nmt_nodes=0, # 0 for no normalisation, 1 for just nmt pairs, and 2 for all pairs
                 require_abs_impurity_redn=True,
                 split_criterion='incomp_side_has_pts', #'both_sides_have_pts',incomp_side_has_pts 'all_splits_ok'
                 split_class='parent_class', # 'contained_pts_class' parent_class
                 split_weight='hybrid_prob_empirical',#parent_weight', # 'contained_pts_weight' parent_weight
                 min_split_weight=1.,
                 base_tree_algo='scikit', # scikit or isotree: NOTE: gini_l1 criterion not available for scikit (gini willl be used)
                 simplify=True,
                 min_split_weight_type='num_pts' # 'prop_of_N' or 'num_pts'
                 ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.presort = presort
        self.split_pts=split_pts
        self.n_features_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.simplify=simplify
        self.tree_ = None
        
        self.feat_data_types=feat_data_types
        self.incr_feats=incr_feats
        self.decr_feats=decr_feats
        self.monotonicity_type=monotonicity_type
        self.monotonicity_params=monotonicity_params
        self.normalise_nmt_nodes=normalise_nmt_nodes
        self.require_abs_impurity_redn=require_abs_impurity_redn
        self.complexity_pruned_trees=None
        self.complexity_pruned_alphas=None
        self.split_criterion=split_criterion
        self.split_class=split_class
        self.split_weight=split_weight
        self.min_split_weight=min_split_weight
        self.base_tree_algo=base_tree_algo
        self.min_split_weight_type=min_split_weight_type
        self.univar_vals=None
        self.univar_vals_num=None
        self.univar_probs=None
        #print(self.require_abs_impurity_redn)
        #print(self.order_ambiguity_weight_R)
    def copy(self):
        return deepcopy(self)
    
    def set_tree(self,tree_):
        self.tree_=tree_
    def create_complexity_pruned_trees(self,alpha_max=None):
        if alpha_max==None:
            alpha_max=np.inf
        # add current trees
        alphas=[0.]
        trees=[self.tree_.copy()]
        # find all pruned trees in order of increasing penlayt on #leaves
        tree_i=self.tree_
        abort=False
        # calculate resubst errors
        tree_i.number_nodes(calc_resubs_err=True)
        
        while len(tree_i.leaf_nodes)>1 and not abort:
            # calculate increase in resubst error for branch nodes to find min
            min_incr=np.inf
            branches_to_fuse=[]
            i_branch=0
            for n in tree_i.branch_nodes:
                incr_resubst=(n.resubst_err_node-n.resubst_err_branch)/(n.num_leaves_branch-1)
                if incr_resubst<min_incr:
                    branches_to_fuse=[i_branch]
                    min_incr=incr_resubst
                elif incr_resubst==min_incr:
                    branches_to_fuse.append(i_branch)   
                i_branch=i_branch+1
            if min_incr>alpha_max:
                abort=True
            else: # continue
                # craete next tree and add
                tree_i_plus_1=tree_i.copy()
                
                for i in branches_to_fuse:
                    tree_i_plus_1.branch_nodes[i].fuse()
                tree_i_plus_1.number_nodes()
                tree_i_plus_1.peak_leaves  =len(tree_i_plus_1.leaf_nodes) 
                alphas.append(min_incr)
                trees.append(tree_i_plus_1)
                tree_i=tree_i_plus_1
        if self.monotonicity_type=='ict':
            for it in np.arange(len(trees)):
                self.fit_monotone_ICT(self.incr_feats,self.decr_feats,normalise_nmt_nodes=self.normalise_nmt_nodes,tree_=trees[it],split_criterion=self.split_criterion, split_class=self.split_class,split_weight=self.split_weight,min_split_weight=self.min_split_weight_pts)        
                
        self.complexity_pruned_alphas=alphas
        self.complexity_pruned_trees=trees
        return [self.complexity_pruned_alphas,self.complexity_pruned_trees]
#    def copy_tree(tree,reset_peak_leaves=True):
#        tree_=tree.copy()
#        if reset_peak_leaves:
#            tree_.peak_leaves=0
#        return tree_
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a decision tree from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression). In the regression case, use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """
        
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=None, accept_sparse=False)
            y = check_array(y, ensure_2d=False, dtype=None)
            

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        
        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        

        check_classification_targets(y)
        y = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        if self.class_weight is not None:
            y_original = np.copy(y)

        y_encoded = np.zeros(y.shape, dtype=np.int)

        classes_k, y_encoded[:, 0] = np.unique(y[:, 0],return_inverse=True)
        self.classes_=classes_k
        self.n_classes_=classes_k.shape[0]
        
        y = y_encoded

        if self.class_weight is not None:
            expanded_class_weight = compute_sample_weight(self.class_weight, y_original)


        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

#        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
#            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(np.ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(np.ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not (0. < self.min_samples_split <= 1. or
                2 <= self.min_samples_split):
            raise ValueError("min_samples_split must be in at least 2"
                             " or in (0, 1], got %s" % min_samples_split)
        if not (0. < self.min_samples_leaf <= 0.5 or
                1 <= self.min_samples_leaf):
            raise ValueError("min_samples_leaf must be at least than 1 "
                             "or in (0, 0.5], got %s" % min_samples_leaf)

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
#            if (getattr(sample_weight, "dtype", None) != DOUBLE or
#                    not sample_weight.flags.contiguous):
#                sample_weight = np.ascontiguousarray(
#                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than or equal "
                             "to 0")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if self.presort == 'auto':
            presort = True

        # If multiple trees are built on the same dataset, we only want to
        # presort once. Splitters now can accept presorted indices if desired,
        # but do not handle any presorting themselves. Ensemble algorithms
        # which desire presorting must do presorting themselves and pass that
        # matrix into each tree.
        if X_idx_sorted is None and presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        if presort and X_idx_sorted.shape != X.shape:
            raise ValueError("The shape of X (X.shape = {}) doesn't match "
                             "the shape of X_idx_sorted (X_idx_sorted"
                             ".shape = {})".format(X.shape,
                                                   X_idx_sorted.shape))

        # Build tree
        criterion = self.criterion
#        if not isinstance(criterion, Criterion):
#            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
#                                                         self.n_classes_)
#        if not callable(criterion):
#            criterion=CRITERIA_CLF[self.criterion]
        
#        splitter = self.splitter
#        if not isinstance(self.splitter, Splitter):
#            splitter = SPLITTERS[self.splitter](criterion,
#                                                self.max_features_,
#                                                min_samples_leaf,
#                                                min_weight_leaf,
#                                                random_state,
#                                                self.presort)

        #self.tree_ = DecisionTree(self.n_features_, self.n_classes_)
        # this is a very basic but inefficient way of implementing class weights
        train_X=X.copy()
        train_y=y.copy()
        train_y=np.ravel(train_y)
        if  sample_weight is None:
            sample_weight=np.ones(len(train_y))
        self.sample_weight=sample_weight
        sample_weight_sum=np.sum(self.sample_weight)
        if self.min_split_weight_type=='num_pts':
            self.min_split_weight_pts=self.min_split_weight
        else:
            self.min_split_weight_pts=self.min_split_weight*sample_weight_sum
            
#        if sample_weight is not None:
#            #X=np.zeros(X.shape,dtype='float')-99.
#            ttl_weights=np.sum(sample_weight)
#            curr_row=0
#            for i in np.arange(X.shape[0]):
#                weight=np.int(sample_weight[i]/ttl_weights*X.shape[0])
#                train_X[curr_row:curr_row+ weight,:]=X[i,:]
#                train_y[curr_row:curr_row+ weight,:]=y[i,:]
#                curr_row=curr_row+weight
        # normalise_nmt_nodes=False
        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0: # ie no limit on leaf nodes
            if self.base_tree_algo=='scikit':
                if self.criterion=='gini_l1':
                    #print('Warning: gini_l1 criterion is not available with scikit base treee algorithm. It has been changed to gini')
                    self.criterion='gini'
                self.train_X=train_X.astype(np.float64,order='C')
                self.train_y=train_y
                scikit_dt= DecisionTreeClassifier(self.criterion,
                                           splitter=self.splitter,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                           max_features=self.max_features, 
                                           random_state=self.random_state,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           min_impurity_decrease=self.min_impurity_decrease,
                                           class_weight=self.class_weight,
                                           presort=self.presort)
                scikit_dt.fit(train_X,train_y,self.sample_weight)
#                std_tree = DecisionTree(train_X,train_y,self.criterion,
#                                    feat_data_types=self.feat_data_types,
#                                    split_pts=self.split_pts,
#                                    classes=np.sort(np.unique(y)),
#                                    sample_weight=self.sample_weight,
#                                    incr_feats=None,
#                                    decr_feats=None)
#                self.tree_=extract_scikit_tree(scikit_dt,std_tree )
                non_zero_sample_weights=self.sample_weight>0
                self.tree_array=DecisionTreeArray(scikit_dt.tree_,
                                              X.shape[1],
                                             self.n_classes_, 
                                             self.incr_feats,
                                             self.decr_feats,
                                             self.train_X[non_zero_sample_weights,:],
                                             y_encoded[non_zero_sample_weights],#train_y,
                                             self.sample_weight[non_zero_sample_weights],
                                             allow_extra_nodes=0 if self.min_split_weight_pts==0 else np.int32(np.sum(self.sample_weight)/self.min_split_weight_pts),
                                             normalise_nmt_nodes=self.normalise_nmt_nodes,
                                             split_criterion=self.split_criterion, 
                                             split_class=self.split_class,
                                             split_weight=self.split_weight,
                                             min_split_weight=self.min_split_weight_pts)
            else: # use pure python isotree (slow, but has gini_l1)
                raise NotImplemented('Not implemented any more')
#                self.tree_=build_tree_depth_first(train_X,train_y,criterion,self.max_features_,random_state,
#                                            min_samples_split,
#                                            min_samples_leaf,
#                                            min_weight_leaf,
#                                            max_depth, self.min_impurity_decrease,
#                                            self.feat_data_types,
#                                            self.split_pts,
#                                            self.sample_weight,
#                                            incr_feats=None,
#                                            decr_feats=None,
#                                            require_abs_impurity_redn=self.require_abs_impurity_redn)
        else: # limit on number of leaf nodes
            raise NotImplemented('Not implemented yet')
        
        if self.monotonicity_type=='ict':
            #self.fit_monotone_ICT(self.incr_feats,self.decr_feats,normalise_nmt_nodes=self.normalise_nmt_nodes,split_criterion=self.split_criterion, split_class=self.split_class,split_weight=self.split_weight,min_split_weight=self.min_split_weight_pts)
            self.fit_monotone_ICT_2()
            
        # trim final array sizes
        self.tree_array.trim_to_size()
        return self
    
    def fit_monotone_ICT_2(self):
        # prepare for split probabilities
        if self.normalise_nmt_nodes!=0:
            if self.split_weight=='univar_prob_distn' or self.split_weight=='hybrid_prob':#
                distns,vals_nums_arr, vals_arr,probs_arr=calculate_univariate_distns(  self.train_X[self.sample_weight!=0.,:],sample_weights=self.sample_weight[self.sample_weight!=0.])
                #self.univariate_distns=distns
                self.univar_vals=vals_arr
                self.univar_vals_num=vals_nums_arr
                self.univar_probs=probs_arr
            elif self.split_weight=='hybrid_prob_empirical':
                distns,vals_nums_arr, vals_arr,probs_arr=calculate_univariate_distns(self.train_X[self.sample_weight!=0.,:],sample_weights=self.sample_weight[self.sample_weight!=0.],max_discrete_bins=1e8)
                #self.univariate_distns=distns
                self.univar_vals=vals_arr
                self.univar_vals_num=vals_nums_arr
                self.univar_probs=probs_arr

            elif self.split_weight=='prob_empirical_cond':
                raise NotImplemented
#                self.univariate_distns=dict()
#                #self.univariate_distns[-99]=calculate_univariate_distns(train_X[self.sample_weight!=0.,:],sample_weights=self.sample_weight[self.sample_weight!=0.],max_discrete_bins=1e8)
#                real_train_X=self.train_X[self.sample_weight!=0.,:]
#                real_train_y=self.train_y[self.sample_weight!=0.]
#                real_sample_wgts=self.sample_weight[self.sample_weight!=0.]
#                for iclass in self.classes_:
#                    self.univariate_distns[iclass]=calculate_univariate_distns(real_train_X[real_train_y==iclass,:],sample_weights=real_sample_wgts[real_train_y==iclass],max_discrete_bins=1e8)
            else: 
                pass
                #self.univariate_distns=None
        else: 
            pass
            #self.univariate_distns=None
            
        # re-solve nodes to monotonise
        keep_going=True
        self.num_iterations=0
        while keep_going:
            res=self.tree_array.monotonise(univar_vals=self.univar_vals,univar_probs=self.univar_probs,univar_vals_num=self.univar_vals_num )
            self.num_iterations=self.num_iterations+1
            if self.simplify:
                if res>0:
                    num_fuses=self.tree_array.simplify() # fuse nodes with same predicted class
                    while num_fuses>0:                    
                        num_fuses=self.tree_array.simplify()
                    # reset cdfs to data cdfs for neext round of isotonic regression
                    pm_pairs=self.tree_array.get_increasing_leaf_node_pairs()   
                    pm_pairs_clean=self.tree_array.eliminate_unnecessary_incr_pairs(pm_pairs)
                    nmt_pairs=self.tree_array.get_non_monotone_pairs(pm_pairs_clean)
                    if len(nmt_pairs)>0:
                        #reset for the next round of monotonising regression
                        leaf_ids_=self.tree_array.leaf_ids_obj.get_idx_array()
                        self.tree_array.cdf[leaf_ids_,:]=self.tree_array.cdf_data[leaf_ids_,:]
                        self.tree_array.pred_class[leaf_ids_]=np.argmax(self.tree_array.cdf[leaf_ids_,:]>=0.5, axis=1)
                        keep_going=True
                    else:
                        keep_going=False
                else:
                    keep_going=False
                #keep_going=res>0 #and not normalise_nmt_nodes
            else:
                keep_going=False
        #print('FINAL num leaves: ' + str(len(self.tree_.leaf_nodes)))
        return
    
    def fit_monotone_ICT(self,incr_feats, decr_feats,sample_reweights=None,normalise_nmt_nodes=-1,tree_=None,split_criterion='both_sides_have_min_sample_wgt', split_class='parent_class',split_weight='hybrid_prob_empirical',min_split_weight=0.5):

        if normalise_nmt_nodes!=-1: # override default !=self.normalise_nmt_nodes:
            self.normalise_nmt_nodes=normalise_nmt_nodes
        if tree_ is None:
            tree_=self.tree_array
        self.incr_feats=incr_feats.copy()
        self.decr_feats=decr_feats.copy()
        self.split_criterion=split_criterion
        self.split_class=split_class
        self.split_weight=split_weight
        #self.min_split_weight_pts=min_split_weight
        if self.normalise_nmt_nodes!=0:
            if self.split_weight=='univar_prob_distn' or self.split_weight=='hybrid_prob':#
                self.univariate_distns=calculate_univariate_distns(  self.train_X[self.tree_.sample_weight!=0.,:],sample_weights=self.tree_.sample_weight[self.tree_.sample_weight!=0.])
            elif self.split_weight=='hybrid_prob_empirical':
                self.univariate_distns=calculate_univariate_distns(self.train_X[self.sample_weight!=0.,:],sample_weights=self.sample_weight[self.sample_weight!=0.],max_discrete_bins=1e8)
            elif self.split_weight=='hybrid_prob_empirical_orig_train':
                self.univariate_distns=calculate_univariate_distns(self.train_X,max_discrete_bins=1e8)
            elif self.split_weight=='prob_empirical_cond':
                self.univariate_distns=dict()
                #self.univariate_distns[-99]=calculate_univariate_distns(train_X[self.sample_weight!=0.,:],sample_weights=self.sample_weight[self.sample_weight!=0.],max_discrete_bins=1e8)
                real_train_X=self.tree_.train_X[self.tree_.sample_weight!=0.,:]
                real_train_y=self.tree_.train_y[self.tree_.sample_weight!=0.]
                real_sample_wgts=self.sample_weight[self.tree_.sample_weight!=0.]
                for iclass in self.classes_:
                    self.univariate_distns[iclass]=calculate_univariate_distns(real_train_X[real_train_y==iclass,:],sample_weights=real_sample_wgts[real_train_y==iclass],max_discrete_bins=1e8)
            else: 
                self.univariate_distns=None
        else: 
            self.univariate_distns=None
        # if requested, reweight nodes in terms of original samples used to train tree
        #if sample_reweights is not None:
        #    sample_reweights=sample_reweights
        #    sample_reweights=self.sample_weight
#        if not X_reweight_nodes is None:
#            if y_reweight_nodes is None:
#                y_encoded=None
#            else:
#                y_encoded = np.zeros(y_reweight_nodes.shape, dtype=np.int)
#                classes_k, y_encoded = np.unique(y_reweight_nodes,return_inverse=True)
#
#            self.tree_.reweight_nodes(X_reweight_nodes,y_encoded)
        # re-solve nodes to monotonise
        keep_going=True
        tree_.num_iterations=0
        while keep_going:
            #print('ICT iteration started, num leaves: ' + str(len(self.tree_.leaf_nodes))) 
            res=tree_.monotonise(incr_feats, decr_feats,sample_reweights=sample_reweights,normalise_nmt_nodes=self.normalise_nmt_nodes,split_criterion=self.split_criterion, split_class=self.split_class,split_weight=self.split_weight,min_split_weight=self.min_split_weight_pts,univariate_distns=self.univariate_distns)
            tree_.num_iterations=tree_.num_iterations+1
            if self.simplify:
                if res>0:
                    num_fuses=tree_.simplify() # fuse nodes with same predicted class
                    while num_fuses>0:                    
                        num_fuses=tree_.simplify()
                    # reset cdfs to data cdfs for neext round of isotonic regression
                    tree_.cdf[tree_.leaf_ids_obj.get_idx_array(),:]=tree_.cdf_data[tree_.leaf_ids_obj.get_idx_array(),:]
                keep_going=res>0 #and not normalise_nmt_nodes
            else:
                keep_going=False
        #print('FINAL num leaves: ' + str(len(self.tree_.leaf_nodes)))
        return
        
    
    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.tree_array is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

#        if check_input:
#            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
        if len(X.shape)<2:
            X=X.reshape([1,len(X)])
        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        X = self._validate_X_predict(X, check_input)
        cum_prob = self.predict_cum_proba(X)
        #proba = self.predict_proba(X)
        #cum_prob=np.ones(proba.shape)
        #cum_prob[:,0]=proba[:,0]
        #for i in np.arange(1,proba.shape[1]-1):
        #    cum_prob[:,i]=cum_prob[:,i-1]+proba[:,i]
        #return self.classes_.take(np.argmax(cum_prob>=0.5, axis=1), axis=0)
        return self.classes_.take(np.argmax(cum_prob>=0.5, axis=1), axis=0)
    def predict_complexity_sequence(self, X, check_input=True):
        X = self._validate_X_predict(X, check_input)
        
        results=np.zeros([X.shape[0],len(self.complexity_pruned_trees)])
        for  i in np.arange(len(self.complexity_pruned_trees)):
            proba = self.predict_proba(X,tree_=self.complexity_pruned_trees[i])
            results[:,i]=self.classes_.take(np.argmax(proba, axis=1), axis=0)
        return results 
        
    def predict_proba(self, X, check_input=True,tree_=None):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Parameters
        ----------
        X : array-like or sparse matrix of shatree_=Nonepe = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
       # if tree_==None:
       #     tree_=self.tree_
        X = self._validate_X_predict(X, check_input)
        #proba = tree_.predict_proba(X)
        proba = self.tree_array.predict_proba(X)
        return proba #self.classes_.take(np.argmax(proba, axis=1), axis=0)
    def predict_cum_proba(self, X, check_input=True,tree_=None):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Parameters
        ----------
        X : array-like or sparse matrix of shatree_=Nonepe = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
       # if tree_==None:
       #     tree_=self.tree_
        X = self._validate_X_predict(X, check_input)
        #proba = tree_.predict_proba(X)
        proba = self.tree_array.predict_cum_proba(X)
        return proba #self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_array.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree
        .. versionadded:: 0.18
        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        indicator : sparse csr array, shape = [n_samples, n_nodes]
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    @property
    def feature_importances_(self):
        """Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.tree_ is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        return self.tree_.compute_feature_importances()
        
def build_tree_depth_first(X,y,criterion,max_features_,random_state,min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth, min_impurity_decrease,
                                        feat_data_types,split_pts,sample_weight,
                                        incr_feats=None,
                                        decr_feats=None,require_abs_impurity_redn=False,
                                        split_criterion=None, split_class=None,split_weight=None,min_split_weight=1.,
                                        univariate_distns=None): 
   classes=np.sort(np.unique(y))
   num_classes=len(classes)
   num_feats=X.shape[1]
   tree = DecisionTree(X,y,criterion,feat_data_types=feat_data_types,split_pts=split_pts,classes=classes,sample_weight=sample_weight,incr_feats=incr_feats,decr_feats=decr_feats,split_criterion=split_criterion, split_class=split_class,split_weight=split_weight,min_split_weight=min_split_weight,univariate_distns=univariate_distns)
   tree.grow_node(None,max_features_,random_state,min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth,sample_weight,require_abs_impurity_redn=require_abs_impurity_redn)
   tree.number_nodes()       
   tree.peak_leaves  =len(tree.leaf_nodes) 
   return tree
def normcdf_std(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0 
def normcdf(x,mean_,stdev_):
    #'Cumulative distribution function for the standard normal distribution'
    return normcdf_std((x-mean_)/stdev_)
    
def calculate_univariate_distns(X, sample_weights=None,
                                max_discrete_bins=10,
                                max_resolution=500.):
    if sample_weights is None:
        sample_weights=np.ones(X.shape[0])
    sample_wgt_sum=np.sum(sample_weights)
    num_feats=X.shape[1]
    distns=dict()
    vals_arr=np.zeros([num_feats,np.int32(1e5)+2],dtype=np.float64)
    probs_arr=np.zeros([num_feats,np.int32(1e5)+2],dtype=np.float64)
    vals_nums_arr=np.zeros(num_feats,dtype=np.int32)
    for i in np.arange(num_feats):
        X_i=X[:,i]
        uniq_vals=np.sort(np.unique(X_i))
        if len(uniq_vals)<=max_discrete_bins:
            vals=uniq_vals
            if np.sum(vals.astype(np.int32)-vals)==0:
                probs=np.bincount(X_i.astype(np.int32),sample_weights )
                probs=probs[probs!=0]
                probs=probs/sample_wgt_sum
#                if len(probs)!=len(vals):
#                    print('sdfsdf')
            else:
                ttls=dict()
                for k in vals:
                    ttls[k]=0.
                for kk in np.arange(X.shape[0]):
                    ttls[X_i[kk]]=ttls[X_i[kk]]+sample_weights[kk]
                probs=np.zeros(len(vals),dtype=np.float64)
                k_i=0
                for kk in vals:
                    probs[k_i]=ttls[kk]/sample_wgt_sum
                    k_i=k_i+1
                    
            #probs=[np.sum(sample_weights[X_i==v])/sample_wgt_sum for v in vals]
        else: # treat as continuous and fit normal curve
            mean_=np.dot(X_i,sample_weights)/sample_wgt_sum# np.mean(X_i)
            stdev_=np.sqrt(np.dot((X_i-mean_)**2,sample_weights)/(sample_wgt_sum))
            minXi=np.min(uniq_vals)
            maxXi=np.max(uniq_vals)
            interval=(maxXi-minXi)/max_resolution
            vals=list(np.arange(minXi,maxXi+interval,interval))
            vals=[-np.inf] + vals + [np.inf]
            probs=[normcdf(vals[j+1],mean_,stdev_)-normcdf(vals[j],mean_,stdev_) for j in np.arange(len(vals))[0:-1]]
            vals_centre=[np.mean([vals[j],vals[j+1]]) for j in np.arange(len(vals))[0:-1]]
            vals=np.asarray(vals_centre,dtype=np.float64)
            
        distns[i]=[vals,np.asarray(probs)]
        n_vals=len(vals)
        vals_nums_arr[i]=n_vals
        vals_arr[i,0:n_vals]=vals
        probs_arr[i,0:n_vals]=probs
    vals_arr=vals_arr[:,0:n_vals]
    probs_arr=probs_arr[:,0:n_vals]
    return distns,vals_nums_arr, vals_arr,probs_arr

def calc_probability(dist_vals,dist_probs,min_val,max_val):
    if min_val==-np.inf:
        return np.sum(dist_probs[dist_vals<=max_val])
    elif max_val==np.inf:
        return np.sum(dist_probs[dist_vals>min_val])
    else:
        return np.sum(dist_probs[np.logical_and(dist_vals>min_val,dist_vals<=max_val)])


        
        
class int_reference:
    def __init__(self,intt):
       self.value=intt
    def change(self,intt):
       self.value=intt 
    def increment(self):
        self.value=self.value+1
## returns a sstructure that mocks the scikit learn tree structure
#def get_mock_scikit_tree(bespoke_tree):
#    class mock_scikit_tree:
#        def __init__(self,value,feature,threshold,children_left,children_right):
#            self.value=value
#            self.feature=feature
#            self.threshold=threshold
#            self.children_left=children_left
#            self.children_right=children_right
#            self.node_count=len(feature)
#            
#    value=[]#list_([])
#    feature=[]
#    threshold=[]
#    children_left=[]
#    children_right=[]
#    id_=int_reference(0)
#    def traverse_node(node=None):
#        if node is None:
#            node=bespoke_tree.root_node
#        if node.is_leaf():
#            value.append([node.probabilities])
#            feature.append(None)#feature+[None]
#            threshold.append(None)#=threshold+[None]
#            children_left.append(TREE_LEAF)#=children_left+[TREE_LEAF]
#            children_right.append(TREE_LEAF)#=children_right+[TREE_LEAF]
#            #id_.increment()
#        else:
#            value.append([node.probabilities])
#            feature.append(node.decision_feat-1)
#            threshold.append(node.decision_values)
#            #children_left.append(id_.value+1)
#            #children_right.append(id_.value+1)
#            id_.increment()#=id_+1
#            curr_index=len(children_left)
#            children_left.append(id_.value)
#            children_right.append(-99)
#            traverse_node(node.left)
#            id_.increment()
#            children_right[curr_index]=id_.value#.append(id_.value)
#            traverse_node(node.right)
#    traverse_node()
#    return mock_scikit_tree(value,feature,threshold,children_left,children_right)

        