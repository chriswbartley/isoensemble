# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:03:20 2016
@author: 19514733
"""
from calc_mcc_interp import calc_mcc_interp
from sklearn import tree#.DecisionTreeClassifier()
#from  pmtree import PMDecisionTreeClassifier #pmtree_preISORFreview pmtree
from  isoensemble import IsoDecisionTreeClassifier

from expset_reader import expset_reader 
import pm_misc

import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn.metrics import  confusion_matrix

root=pm_misc.get_root_dir()#'\\uniwa.uwa.edu.au\\userhome\\Students3\\19514733\\My Documents\\Experiments_2\\PM\\'
#run_id=7 #'haberman','ljubBC','pima','cleve_hrt','SAheart','germancred','carALT','autompgALT'5
expset_id=5
datasets=['SAheart','cleve_hrt','ljubBC','pima','german','WBCdiag','autompgALT','carALT','haberman']
datasets=['carALT'] #'carALT',
subsamples=[100]#,100]#,200]

max_features=None
#run_variants=['svmB0-mcsm0-00-unc0-0000','svmB0-mcsm0-00-ncm0-dRs0','svmB0-mcsm0-00-ncr0-dRs0'] 
#run_var_spec='svmB0-mcsm0-00-ads3-dB00'#'svmB0-mcsm0-00-unc0-0000'  #'rfcB0-xwgt1-00-ads2-dRr1' 'rfcB0-xwgt0-00-unc0-0000','rfcB0-xwgt0-00-ncm0-dRs0' ,'rfcB0-xwgt0-00-ncr0-dRs0'
constr_set='db00'
rand_state=99
#run_var_spec='svmB0-mcsm0-00-unc0-0000'
tot=0.
numExpts=1
tree_alpha_penalties=np.arange(0,0.5,0.0001)
mt_type='ict'
require_abs_impurity_redn=True
normalise_nmt_nodes=0
split_criterion='both_sides_have_min_sample_wgt' #'both_sides_have_pts',incomp_side_has_pts 'all_splits_ok'
split_class='parent_class' # 'contained_pts_class' parent_class
split_weight='hybrid_prob_empirical' # 'contained_pts_weight' parent_weight  univar_prob_distn              
min_split_weight=0.25 

for dataset_orig in datasets:
        for ss in subsamples:
            dataset=(dataset_orig + '-'+ str(ss))# if append_ss else    dataset_orig;
            print(dataset)
            accs_all=np.zeros([numExpts,8])
            kappas_all=np.zeros([numExpts,8])
            times_all=np.zeros([numExpts,8])
            num_leaves_all=np.zeros([numExpts,8])
            for iexpt in  np.arange(numExpts)+1:
                #rr=run_reader(root,run_id)
                #rv=run_variant(rr,run_var_spec)
                #expset_id=rr.expset_id
                #model=rv.get_model(dataset,iexpt,M=0 if rv.isunconstrained() else 1)
                #pmrf=model.clf
                print('Starting Expt ' + str(iexpt))
                er=expset_reader(root,expset_id)
                [x_test,y_test]=er.get_data(dataset,iexpt,'00','test')
                [x_train,y_train]=er.get_data(dataset,iexpt,'00','train')
                cvfolds= er.get_cv_folds(dataset,iexpt)
                mfs=er.get_mono_feat_set(dataset,constr_set,iexpt)
                incr_feats=list(mfs[0])
                decr_feats=list(mfs[1])
                mono_feats=incr_feats.copy()
                for i in decr_feats:
                    mono_feats.append(i) 
                #MCs=[]
                # predict previous matlab model
#                #pred_y_prev= model.predict(x_test)#  pmrf.rfc.predict(x)
#            
                
                start=time.time()
                feat_data_types='auto'
                if dataset_orig in ['carALT'] :
                    feat_data_types=['ordinal','ordinal','ordinal','ordinal','ordinal','ordinal']
                    
                ##### BASE PMTREE - NON MONOTONE ICT #######
                clf_mydt=IsoDecisionTreeClassifier(criterion='gini_l1',random_state=rand_state+iexpt,feat_data_types=feat_data_types,max_features=max_features,monotonicity_type=None,require_abs_impurity_redn=require_abs_impurity_redn)#,require_abs_impurity_redn=require_abs_impurity_redn) #max_features=1                
                # use cost_complexity pruning & CV to find best alpha complexity penalty
                best=[0.,-1]
                cv_accs=np.zeros([np.max(cvfolds),len(tree_alpha_penalties)])
                for icv in np.arange(np.max(cvfolds)):
                    cv_train_X=x_train[cvfolds!=icv+1,:]
                    cv_train_y=y_train[cvfolds!=icv+1]
                    cv_test_X=x_train[cvfolds==icv+1,:]
                    cv_test_y=y_train[cvfolds==icv+1]
                    clf_mydt.fit(cv_train_X,np.ravel(cv_train_y)) 
                    [complexity_pruned_alphas,complexity_pruned_trees]=clf_mydt.create_complexity_pruned_trees()
                    pred_y_cv_allcomplexities=clf_mydt.predict_complexity_sequence(cv_test_X)
                    for i_complexity in np.arange(len(complexity_pruned_alphas)):
                        alpha_start=complexity_pruned_alphas[i_complexity]
                        if i_complexity<len(complexity_pruned_alphas)-1:
                            alpha_finish=complexity_pruned_alphas[i_complexity+1]
                        else:
                            alpha_finish=np.inf
                        pred_y_cv=pred_y_cv_allcomplexities[:,i_complexity]
                        cm=confusion_matrix(cv_test_y,pred_y_cv)
                        indxs=np.logical_and(tree_alpha_penalties>=alpha_start,tree_alpha_penalties<alpha_finish)
                        cv_accs[icv,indxs]=(cm[0,0]+cm[1,1])/np.sum(cm)
                accs_alphas=np.mean(cv_accs,0) 
                mx_indxs=np.arange(len(accs_alphas))[accs_alphas==np.max(accs_alphas)]
                mx_indx=np.max(mx_indxs)
                mx_opt_alpha=tree_alpha_penalties[mx_indx]
                # fit final model
                clf_mydt.fit(x_train, y_train) 
                [complexity_pruned_alphas,complexity_pruned_trees]=clf_mydt.create_complexity_pruned_trees(alpha_max=mx_opt_alpha)
                clf_mydt.set_tree(complexity_pruned_trees[len(complexity_pruned_alphas)-1])
                y_pred_mydt=clf_mydt.predict(x_test)
                print('Opt Alpha: ' + str(mx_opt_alpha) + 'leaves:' + str(len(clf_mydt.tree_.leaf_nodes)))
                durn_curr=time.time()-start
                [accuracy, sensitivity, NegPredValue, precision, fMeasure, kappa]=pm_misc.getPerfFromConfusionMat(confusion_matrix(y_test,y_pred_mydt))
                acc_curr= accuracy#np.sum(np.ravel(y_pred_mydt)==np.ravel(y_test))/len(y_test)
                kappa_curr=kappa
                #[complexity_pruned_alphas,complexity_pruned_trees]=clf_mydt.create_complexity_pruned_trees()
                
                #print('resubst err:' + str(clf_mydt.tree_.root_node.resubst_err_node))
                # print mccs
                mccs=np.zeros(len(mono_feats))
                i=0
                for ifeat in mono_feats:
                    mcc=calc_mcc_interp(x_train,clf_mydt,ifeat, [],'od_off' )
                    mcc=mcc[0]
                    #[NoChangeInEitherDirn , MonotoneIncreasing,MonotoneDecreasing, MonotoneIncrDec ]
                    mccs[i]=mcc[0]+0.5*mcc[3] + (mcc[1] if ifeat in incr_feats else mcc[2])
                    i=i+1
                ##### BASE PMTREE - ORDER AMBIGUITY BASED #######
                clf_oa=IsoDecisionTreeClassifier(criterion='entropy',random_state=rand_state+iexpt,feat_data_types=feat_data_types,max_features=max_features,monotonicity_type='order_ambiguity',incr_feats=incr_feats,decr_feats=decr_feats,order_ambiguity_weight_R=1,order_ambiguity_weight_R_Random_Limit=None,require_abs_impurity_redn=require_abs_impurity_redn)#,require_abs_impurity_redn=require_abs_impurity_redn) #max_features=1
                clf_oa.fit(x_train, y_train) 
                y_pred_oa=clf_oa.predict(x_test)
                [accuracy, sensitivity, NegPredValue, precision, fMeasure, kappa]=pm_misc.getPerfFromConfusionMat(confusion_matrix(y_test,y_pred_oa))
                durn_curr=time.time()-start
                acc_oa=accuracy #np.sum(np.ravel(y_pred_oa)==np.ravel(y_test))/len(y_test)
                kappa_oa=kappa
                # print mccs
                mccs_oa=np.zeros(len(mono_feats))
                i=0
                for ifeat in mono_feats:
                    mcc=calc_mcc_interp(x_train,clf_oa,ifeat, [],'od_off' )
                    mcc=mcc[0]
                    #[NoChangeInEitherDirn , MonotoneIncreasing,MonotoneDecreasing, MonotoneIncrDec ]
                    mccs_oa[i]=mcc[0]+0.5*mcc[3] + (mcc[1] if ifeat in incr_feats else mcc[2])
                    i=i+1
                #print(np.mean(mccs_oa))
                ##### SK LEARN TREE #######
                start=time.time()
                clf_sk = tree.DecisionTreeClassifier(criterion='gini',random_state=rand_state+iexpt,max_features=max_features)
                clf_sk.fit(x_train, y_train) 
                y_pred_sklearn=clf_sk.predict(x_test)
                durn_sklearn=time.time()-start
                [accuracy, sensitivity, NegPredValue, precision, fMeasure, kappa]=pm_misc.getPerfFromConfusionMat(confusion_matrix(y_test,y_pred_sklearn))
                
                acc_sklearn=accuracy#np.sum(np.ravel(y_pred_sklearn)==np.ravel(y_test))/len(y_test)
                kappa_sklearn=kappa
                ##### BASE PMTREE - ICT MONOTONE #######
                #clf_iso=PMDecisionTreeClassifier(criterion='gini_l1',random_state=rand_state+iexpt,feat_data_types=feat_data_types,incr_feats=incr_feats,decr_feats=decr_feats,max_features=max_features,monotonicity_type='ict',normalise_nmt_nodes=1)#,require_abs_impurity_redn=require_abs_impurity_redn) #max_features=1                
                # use cost_complexity pruning & CV to find best alpha complexity penalty
          
                best=[0.,-1]
                cv_accs=np.zeros([np.max(cvfolds),len(tree_alpha_penalties)])
                for icv in np.arange(np.max(cvfolds)):
                    cv_train_X=x_train[cvfolds!=icv+1,:]
                    cv_train_y=y_train[cvfolds!=icv+1]
                    cv_test_X=x_train[cvfolds==icv+1,:]
                    cv_test_y=y_train[cvfolds==icv+1]
                    clf_iso=IsoDecisionTreeClassifier(criterion='gini_l1',random_state=rand_state+iexpt,feat_data_types=feat_data_types,incr_feats=incr_feats,decr_feats=decr_feats,max_features=max_features,monotonicity_type=None,normalise_nmt_nodes=0,require_abs_impurity_redn=require_abs_impurity_redn)#,require_abs_impurity_redn=require_abs_impurity_redn) #max_features=1                
                    clf_iso.fit(cv_train_X,np.ravel(cv_train_y)) 
                    clf_iso.monotonicity_type=mt_type
                    clf_iso.normalise_nmt_nodes=normalise_nmt_nodes
                    clf_iso.split_criterion=split_criterion
                    clf_iso.split_class=split_class
                    clf_iso.split_weight=split_weight
                    clf_iso.min_split_weight=min_split_weight
                    [complexity_pruned_alphas,complexity_pruned_trees]=clf_iso.create_complexity_pruned_trees()
                    pred_y_cv_allcomplexities=clf_iso.predict_complexity_sequence(cv_test_X)
                    for i_complexity in np.arange(len(complexity_pruned_alphas)):
                        alpha_start=complexity_pruned_alphas[i_complexity]
                        if i_complexity<len(complexity_pruned_alphas)-1:
                            alpha_finish=complexity_pruned_alphas[i_complexity+1]
                        else:
                            alpha_finish=np.inf
                        pred_y_cv=pred_y_cv_allcomplexities[:,i_complexity]
                        cm=confusion_matrix(cv_test_y,pred_y_cv)
                        indxs=np.logical_and(tree_alpha_penalties>=alpha_start,tree_alpha_penalties<alpha_finish)
                        cv_accs[icv,indxs]=(cm[0,0]+cm[1,1])/np.sum(cm)
                accs_alphas=np.mean(cv_accs,0) 
                mx_indxs=np.arange(len(accs_alphas))[accs_alphas==np.max(accs_alphas)]
                mx_indx=np.max(mx_indxs)
                mx_opt_alpha=tree_alpha_penalties[mx_indx]
                # fit final model
                clf_iso=IsoDecisionTreeClassifier(criterion='gini_l1',random_state=rand_state+iexpt,feat_data_types=feat_data_types,incr_feats=incr_feats,decr_feats=decr_feats,max_features=max_features,monotonicity_type=None,normalise_nmt_nodes=0)#,require_abs_impurity_redn=require_abs_impurity_redn) #max_features=1                
                clf_iso.fit(x_train,y_train) 
                clf_iso.monotonicity_type=mt_type
                clf_iso.normalise_nmt_nodes=normalise_nmt_nodes 
                clf_iso.split_criterion=split_criterion
                clf_iso.split_class=split_class
                clf_iso.split_weight=split_weight
                clf_iso.min_split_weight=min_split_weight
                #clf_iso.fit(x_train, y_train) 
                [complexity_pruned_alphas,complexity_pruned_trees]=clf_iso.create_complexity_pruned_trees(alpha_max=mx_opt_alpha)
                clf_iso.set_tree(complexity_pruned_trees[len(complexity_pruned_alphas)-1])
                y_pred_iso=clf_iso.predict(x_test)
                print('Opt Alpha ISO: ' + str(mx_opt_alpha) + 'leaves:' + str(len(clf_iso.tree_.leaf_nodes)))
                durn_iso=time.time()-start
                [accuracy, sensitivity, NegPredValue, precision, fMeasure, kappa]=pm_misc.getPerfFromConfusionMat(confusion_matrix(y_test,y_pred_iso))
                acc_iso=accuracy#np.sum(np.ravel(y_pred_iso)==np.ravel(y_test))/len(y_test)
                kappa_iso=kappa
                
#                clf=clf_mydt.copy()
#                clf.fit_monotone_ICT(incr_feats,decr_feats,normalise_nmt_nodes=normalise_nmt_nodes)
#                y_pred_monodt=clf.predict(x_test)
#                acc_monodt=np.sum(np.ravel(y_pred_monodt)==np.ravel(y_test))/len(y_test)
#                [complexity_pruned_alphas,complexity_pruned_trees]=clf.create_complexity_pruned_trees()
                
                # calc Mccs
                if False:
                    mccs_pm=np.zeros(len(mono_feats))
                    i=0
                    for ifeat in mono_feats:
                        mcc=calc_mcc_interp(x_train,clf_iso,ifeat, [],'od_off' )
                        mcc=mcc[0]
                        #[NoChangeInEitherDirn , MonotoneIncreasing,MonotoneDecreasing, MonotoneIncrDec ]
                        mccs_pm[i]=mcc[0]+0.5*mcc[3] + (mcc[1] if ifeat in incr_feats else mcc[2])
                        i=i+1
    
                    if np.mean(mccs_pm)<1.0:
                        print('ERROR: mcc dt: ' + str(np.mean(mccs))+ 'mcc pmdt: ' + str(np.mean(mccs_pm)))
                    if np.mean(mccs_oa)<1.0:
                        print('OA MCC: mcc dt: ' + str(np.mean(mccs))+ 'mcc oa dt: ' + str(np.mean(mccs_oa)))
                
                # gather results
                acc_prev=0 #np.sum(np.ravel(pred_y_prev)==np.ravel(y_test))/len(y_test)
                kappa_prev=0
                accs_all[iexpt-1,:]=[acc_sklearn,acc_prev, acc_curr, acc_curr-acc_sklearn,acc_iso,acc_iso-acc_curr,acc_oa,acc_oa-acc_curr]
                kappas_all[iexpt-1,:]=[kappa_sklearn,kappa_prev, kappa_curr, kappa_curr-kappa_sklearn,kappa_iso,kappa_iso-kappa_curr,kappa_oa,kappa_oa-kappa_curr]
                times_all[iexpt-1,0:4]=[durn_sklearn,0, durn_curr, durn_curr-durn_sklearn]
                num_leaves_all[iexpt-1,0:4]=[(clf_sk.tree_.node_count+1.)/2., len(clf_mydt.tree_.leaf_nodes), len(clf_iso.tree_.leaf_nodes), len(clf_oa.tree_.leaf_nodes)]
                #comparison_acc=[acc_sklearn,acc_prev, acc_curr, acc_curr-acc_sklearn,acc_monodt,acc_monodt-acc_curr]
                
                #print('      Acc comparison: ' + str(comparison_acc))

#                # EXPORT TREE FROM SKLEARN
                if False:
                    clf_mydt.tree_.printtree()
                    import pydotplus
                    import io
                    dot_data = io.StringIO()
                    tree.export_graphviz(clf_sk, out_file=dot_data)
                    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                    graph.write_pdf(root + 'reports/trees/' +  dataset + ".pdf")


            print(np.mean(accs_all,0))
            print(np.mean(kappas_all,0))
            print(np.mean(num_leaves_all,0))
            #print(np.mean(times_all,0))