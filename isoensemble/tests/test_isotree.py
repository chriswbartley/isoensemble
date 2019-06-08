from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
import isoensemble
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier
import time
def load_data_set():
    # Load data

    data = load_boston()
    y = data['target']
    X = data['data']
    features = data['feature_names']
    # Specify monotone features
    incr_feat_names = ['RM']#['RM', 'RAD']
    decr_feat_names = ['CRIM', 'LSTAT'] # ['CRIM', 'DIS', 'LSTAT']
    # get 1 based indices of incr and decr feats
    incr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in incr_feat_names]
    decr_feats = [i + 1 for i in np.arange(len(features)) if
                  features[i] in decr_feat_names]
    # Convert to classification problem
    # Multi-class
    y_multiclass = y.copy()
    thresh1 = 15
    thresh2 = 21
    thresh3 = 27
    y_multiclass[y > thresh3] = 3
    y_multiclass[np.logical_and(y > thresh2, y <= thresh3)] = 2
    y_multiclass[np.logical_and(y > thresh1, y <= thresh2)] = 1
    y_multiclass[y <= thresh1] = 0
    # Binary
    y_binary = y.copy()
    thresh = 21  # middle=21
    y_binary[y_binary < thresh] = -1
    y_binary[y_binary >= thresh] = +1
    return X, y_binary, y_multiclass, incr_feats, decr_feats


# Load data
max_N = 200
np.random.seed(13) # comment out for changing random training set
X, y_binary, y_multiclass, incr_feats, decr_feats = load_data_set()
indx_train=np.random.permutation(np.arange(X.shape[0]))[0:max_N]
inx_test=np.asarray([i for i in np.arange(max_N) if i not in indx_train ])
X_train=X[indx_train,:]
X_test=X[inx_test,:]


y_train=dict()
y_test=dict()
n_classes=dict()
y_train['binary']=y_binary[indx_train]
y_train['multiclass']=y_multiclass[indx_train]
y_test['binary']=y_binary[inx_test]
y_test['multiclass']=y_multiclass[inx_test]
n_classes['binary']=2
n_classes['multiclass']=4

def test_model_fit():
    # Specify hyperparams for model solution
    mt_type='ict'
    require_abs_impurity_redn=True
    max_features=None
    feat_data_types='auto'
    base_tree_algo='scikit' #'scikit' isotree
    normalise_nmt_nodes=0
    min_split_weight=0.25
#    
#    tree_alpha_penalties=np.arange(0,0.5,0.0001)
#    
#    normalise_nmt_nodes=0
#    split_criterion='both_sides_have_min_sample_wgt' #'both_sides_have_pts',incomp_side_has_pts 'all_splits_ok'
#    split_class='parent_class' # 'contained_pts_class' parent_class
#    split_weight='hybrid_prob_empirical' # 'contained_pts_weight' parent_weight  univar_prob_distn              
#    min_split_weight=0.25 

    # Solve models
    acc=dict()
    acc_benchmark={'multiclass-mt': 0.64800000000000002, 'binary-nmt': 0.752, 'binary-mt': 0.81599999999999995, 'multiclass-nmt': 0.61599999999999999}
    num_leaves=dict()
    for response in ['binary','multiclass']: #'multiclass']:#
        y_train_=y_train[response]
        y_test_=y_test[response]
        n_classes_=n_classes[response]
        for constr in ['mt','nmt']:
            clf = isoensemble.IsoDecisionTreeClassifier(criterion='gini_l1',
                                          random_state=11,
                                          feat_data_types=feat_data_types,
                                          max_features=max_features,
                                          monotonicity_type=None if constr=='nmt' else mt_type,
                                          normalise_nmt_nodes=normalise_nmt_nodes,
                                          require_abs_impurity_redn=require_abs_impurity_redn,
                                          incr_feats=incr_feats if constr =='mt' else None,
                                          decr_feats=decr_feats if constr =='mt' else None,
                                          base_tree_algo=base_tree_algo,
                                          min_split_weight=min_split_weight
                                          )

            # Assess fit
            clf.fit(X_train, y_train_)
            #
            y_pred = clf.predict(X_test)
            acc[response + '-' + constr] = np.sum(y_test_ == y_pred) / len(y_test_)
            num_leaves[response + '-' + constr]=len(clf.tree_.leaf_nodes) #[(clf_sk.tree_.node_count+1.)/2., len(clf_mydt.tree_.leaf_nodes), len(clf_iso.tree_.leaf_nodes), len(clf_oa.tree_.leaf_nodes)]
                
            print(acc[response + '-' + constr])
            # Measure monotonicity
            # mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X_test,incr_feats=incr_feats, decr_feats=decr_feats))
    
    print('acc: ' + str(acc))
    print('n leaves: ', str(num_leaves))
    #for key in acc.keys():
    #    npt.assert_almost_equal(acc[key],acc_benchmark[key])
    # print('mcc: ' + str(mcc))
    # npt.assert_almost_equal(clf.oob_score_, 0.85999999999)
    # npt.assert_almost_equal(acc_mc, 0.944999999999)

def benchmark_against_scikit():
    # binary should match, nulti-class could be different because
    # pmsvm  uses montone ensembling but scikit uses OVR.
    #
    # Specify hyperparams for model solution
    require_abs_impurity_redn=True
    max_features=None
    feat_data_types='auto'
    acc=dict()
    num_leaves=dict()
    solve_time=dict()
    # Solve models
    for response in ['multiclass','binary']:#,'multiclass']:
        y_train_=y_train[response]
        y_test_=y_test[response]
        n_classes_=n_classes[response]
        for model in ['isotree','scikit']:
            if model=='isotree':
                clf = isoensemble.IsoDecisionTreeClassifier(criterion='gini', #gini_l1
                                          random_state=11,
                                          feat_data_types=feat_data_types,
                                          max_features=max_features,
                                          monotonicity_type=None ,
                                          normalise_nmt_nodes=0,
                                          require_abs_impurity_redn=require_abs_impurity_redn,
                                          incr_feats= None,
                                          decr_feats= None)
                #clf_iso=clf
                
            else:
                clf = DecisionTreeClassifier(criterion='gini',
                                          random_state=11,
                                          max_features=max_features)
            # Assess fit
            start=time.time()
            clf.fit(X_train, y_train_)
            durn=time.time()-start
            if model=='isotree':
                pred_prob_iso=clf.predict_proba(X_test)
                num_leaves[response + '-' + model]=len(clf.tree_.leaf_nodes) 
                clf.tree_.printtree()
            else:
                pred_prob_sk=clf.predict_proba(X_test)
                num_leaves[response + '-' + model]=(clf.tree_.node_count+1.)/2.

            #
            #test constraints are satisifed
            #res=clf.predict(clf.constraints[0][0,:,1])-clf.predict(clf.constraints[0][0,:,0])
    #            if model=='pmrf':
    #                support_vectors[response + '-' + model]= clf.support_vectors_[0][0,:]
    #                n_support_vectors[response + '-' + model]= np.mean(clf.n_support_)
    #                dual_coef[response + '-' + model]=np.flip(np.sort(np.abs(clf.dual_coef_[0])),axis=0)
    #            else:
    #                support_vectors[response + '-' + model]= clf.support_vectors_[0]
    #                n_support_vectors[response + '-' + model]= np.sum(clf.n_support_[0:n_classes[response]])
    #                dual_coef[response + '-' + model]=np.flip(np.sort(np.abs(clf.dual_coef_[0])),axis=0)
            y_pred = clf.predict(X_test)
            #oob_scores[response + '-' + model] = clf.oob_score_
            solve_time[response + '-' + model]=durn
            acc[response + '-' + model] = np.sum(y_test_ == y_pred) / len(y_test_)
            #oob_scores[response + '-' + model] = clf.oob_score_
            
            #print(acc[response + '-' + model])
            
            # Measure monotonicity
            #mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X,incr_feats=incr_feats, decr_feats=decr_feats))    
    print('acc: ' + str(acc))
    print('Note that performance will vary slightly due to randomisation of feature evaluation, in particular near leaf nodes when to achieve leaf purity there will be multiple complying candidate splits. Trees are high variance classifiers, particularly when grown to leaf purity!')
    print('n leaves: ', str(num_leaves))
        #print(n_support_vectors)
        #print(solve_time)
#    pmsvm_coefs=dual_coef['binary-pmrf']
#    scikit_coefs=dual_coef['binary-scikit']
#    min_len=np.min([pmsvm_coefs.shape[0],scikit_coefs.shape[0]])
#    diff=np.sum(np.abs(scikit_coefs[0:min_len]-pmsvm_coefs[0:min_len]))/np.sum(np.abs(scikit_coefs[0:min_len]))
#    print('dual coef abs diff: ' + str(diff))
#print(support_vectors)
test_model_fit()
#benchmark_against_scikit()