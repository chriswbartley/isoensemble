from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
import isoensemble
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
import time

def get_leaf_counts(pmdtrf):
        numtrees=len(pmdtrf.estimators_) #np.int(self.rfc.get_params()['n_estimators'])
        num_leaves=np.zeros(numtrees,dtype='float')
        for itree in np.arange(numtrees):
            #num_leaves[itree]=len(pmdtrf.estimators_[itree].tree_.leaf_nodes)
            num_leaves[itree]=pmdtrf.estimators_[itree].tree_array.leaf_ids_obj.curr_size
        return num_leaves

def get_peak_leaves(pmdtrf):
        numtrees=len(pmdtrf.estimators_) #np.int(self.rfc.get_params()['n_estimators'])
        num_leaves=np.zeros(numtrees,dtype='float')
        for itree in np.arange(numtrees):
            #num_leaves[itree]=pmdtrf.estimators_[itree].tree_.peak_leaves
            num_leaves[itree]=pmdtrf.estimators_[itree].tree_array.peak_leaves
        return num_leaves
        
def get_num_iterations(pmdtrf):
        numtrees=len(pmdtrf.estimators_) #np.int(self.rfc.get_params()['n_estimators'])
        num_leaves=np.zeros(numtrees,dtype='float')
        for itree in np.arange(numtrees):
            #num_leaves[itree]=pmdtrf.estimators_[itree].tree_.num_iterations
            num_leaves[itree]=pmdtrf.estimators_[itree].num_iterations
        return num_leaves    
      
        
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
max_N = 400
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
    n_estimators = 100#200
    mtry = 3
    mt_type='ict'
    require_abs_impurity_redn=True
    feat_data_types='auto'
    base_tree_algo='scikit' # isotree
    normalise_nmt_nodes=2
    min_split_weight=0.25
    split_criterion='both_sides_have_min_sample_wgt'
    split_class='parent_class'
    split_weight='hybrid_prob_empirical'
    min_split_weight_type='prop_N' #num_pts
    simplify=False
    acc_correct={'multiclass-nmt': 0.752, 
                 'binary-nmt': 0.84799999999999998, 
                 'multiclass-mt': 0.74399999999999999, 
                 'binary-mt': 0.85599999999999998}
    acc_correct_scikit={'multiclass-mt': 0.76800000000000002, 
          'binary-nmt': 0.86399999999999999, 
          'binary-mt': 0.872, 
          'multiclass-nmt': 0.72799999999999998}
    acc=dict()
    oob_score=dict()
    for response in ['multiclass']:#,binary'multiclass']: #'multiclass']:#
        y_train_=y_train[response]
        y_test_=y_test[response]
        n_classes_=n_classes[response]
        for constr in ['mt']:#,'nmt']:
            clf = isoensemble.IsoRandomForestClassifier(n_estimators=n_estimators,
                                          criterion='gini_l1',
                                          random_state=11,
                                          feat_data_types=feat_data_types,
                                          max_features=mtry,
                                          monotonicity_type=None if constr=='nmt' else mt_type,
                                          normalise_nmt_nodes=normalise_nmt_nodes,
                                          require_abs_impurity_redn=require_abs_impurity_redn,
                                          incr_feats=incr_feats if constr =='mt' else None,
                                          decr_feats=decr_feats if constr =='mt' else None,
                                          oob_score=True,
                                          base_tree_algo=base_tree_algo,
                                          min_split_weight=min_split_weight,
                                          min_split_weight_type=min_split_weight_type,
                                          split_criterion=split_criterion,
                                          split_class=split_class,
                                          split_weight=split_weight,
                                          simplify=simplify
                                          )

            # Assess fit
            start=time.time()
            clf.fit(X_train, y_train_)
            solve_durn=time.time()-start
            print('solve took: ' + str(solve_durn) + ' secs')
            #
            y_pred = clf.predict(X_test)
            acc[response + '-' + constr] = np.sum(y_test_ == y_pred) / len(y_test_)
            oob_score[response + '-' + constr]=clf.oob_score_ #[(clf_sk.tree_.node_count+1.)/2., len(clf_mydt.tree_.leaf_nodes), len(clf_iso.tree_.leaf_nodes), len(clf_oa.tree_.leaf_nodes)]
                
            #print(acc[response + '-' + constr])
            print(np.mean(get_peak_leaves(clf)))
            print(np.mean(get_leaf_counts(clf)))
            # Measure monotonicity
            # mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X_test,incr_feats=incr_feats, decr_feats=decr_feats))
    
    print('acc: ' + str(acc))
    print('n oob_score: ', str(oob_score))
    # BENCHMARK binary MT acc: 0.864, time: 25.7secs
    #for key in acc.keys():
    #    npt.assert_almost_equal(acc[key],acc_correct_scikit[key])
    # print('mcc: ' + str(mcc))
    # npt.assert_almost_equal(clf.oob_score_, 0.85999999999)
    # npt.assert_almost_equal(acc_mc, 0.944999999999)

def benchmark_against_scikit():
    # binary should match, nulti-class could be different because
    # pmsvm  uses montone ensembling but scikit uses OVR.
    #
    # Specify hyperparams for model solution
    n_estimators = 200
    mtry = 3
    require_abs_impurity_redn=True
    feat_data_types='auto'
    
    acc=dict()
    oob_score=dict()
    solve_time=dict()
    # Solve models
    for response in ['multiclass','binary']:#,'multiclass']:
        y_train_=y_train[response]
        y_test_=y_test[response]
        n_classes_=n_classes[response]
        for model in ['isotree','scikit']:
            if model=='isotree':
                clf = isoensemble.IsoRandomForestClassifier(n_estimators=n_estimators,
                                          criterion='gini',
                                          random_state=11,
                                          feat_data_types=feat_data_types,
                                          max_features=mtry,
                                          monotonicity_type=None,
                                          normalise_nmt_nodes=0,
                                          require_abs_impurity_redn=require_abs_impurity_redn,
                                          incr_feats=None,
                                          decr_feats=None,
                                          oob_score=True
                                          )
                #clf_iso=clf
                
            else:
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                          criterion='gini',
                                          random_state=11,
                                          max_features=mtry,
                                          oob_score=True)
            # Assess fit
            start=time.time()
            clf.fit(X_train, y_train_)
            durn=time.time()-start

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
            oob_score[response + '-' + model]=clf.oob_score_
            
            #oob_scores[response + '-' + model] = clf.oob_score_
            
            #print(acc[response + '-' + model])
            
            # Measure monotonicity
            #mcc[response + '-' + constr] = np.mean(clf.calc_mcc(X,incr_feats=incr_feats, decr_feats=decr_feats))    
    print('acc: ' + str(acc))
    print('n oob_score: ', str(oob_score))
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
