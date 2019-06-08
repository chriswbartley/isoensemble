"""Forest of trees-based ensemble methods

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.

"""

# Authors: Chris Bartley <chris@bartleys.net>
#
# License: BSD 3 clause

from sklearn.ensemble.forest import ForestClassifier
import isoensemble
import numpy as np


class IsoRandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and use averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_split : float, optional (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. versionadded:: 0.18

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : dict, list of dicts, "balanced",
        "balanced_subsample" or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """
    def __init__(self,
                 n_estimators=10,
                 criterion="gini_l1",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 feat_data_types='auto',
                 incr_feats=None,
                 decr_feats=None,
                 monotonicity_type='ict', #'ict' or 'order_ambiguity'
                 monotonicity_params=None,
                 normalise_nmt_nodes=0, # only for monotonicity_type='ict'
                 require_abs_impurity_redn=False,
                 split_criterion='incomp_side_has_pts', #'both_sides_have_pts',incomp_side_has_pts 'all_splits_ok'
                 split_class='parent_class', # 'contained_pts_class' parent_class
                 split_weight='contained_pts_weight' ,# 'contained_pts_weight' parent_weight
                 min_split_weight=1,
                 base_tree_algo='scikit',
                 min_split_weight_type='num_pts',# 'prop_of_N' or 'num_pts'# scikit or isotree: NOTE: gini_l1 criterion not available for scikit (gini willl be used)
                 simplify=True
                 ): # only for monotonicity_type='order_ambiguity'
        super(IsoRandomForestClassifier, self).__init__(
            base_estimator=isoensemble.IsoDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes", "min_impurity_decrease",
                              "random_state","feat_data_types","incr_feats","decr_feats",
                              "monotonicity_type","monotonicity_params","normalise_nmt_nodes",
                              "require_abs_impurity_redn",
                              "split_criterion", #incomp_side_has_pts", #'both_sides_have_pts',incomp_side_has_pts 'all_splits_ok'
                              "split_class", # 'contained_pts_class' parent_class
                              "split_weight",
                              "min_split_weight",
                              "base_tree_algo",
                              "min_split_weight_type","simplify" # 'prop_of_N' or 'num_pts'
                 ),
                              
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        self.feat_data_types=feat_data_types
        self.incr_feats=incr_feats
        self.decr_feats=decr_feats
        self.monotonicity_type=monotonicity_type
        self.monotonicity_params=monotonicity_params
        self.normalise_nmt_nodes=normalise_nmt_nodes
        self.require_abs_impurity_redn=require_abs_impurity_redn
        self.split_criterion=split_criterion
        self.split_class=split_class 
        self.split_weight=split_weight
        self.min_split_weight=min_split_weight
        self.base_tree_algo=base_tree_algo
        self.min_split_weight_type=min_split_weight_type
        self.simplify=simplify
        
    # fit() is only over-ridden for the sake of order-ambiguity technique, for ict it is the same and does not need to be reimplemented here.
    def fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Returns self.
        """

        # perform standard fit        
        super(IsoRandomForestClassifier,self).fit(X, y, sample_weight)
        #print('wrong version')
        return self    
    
    def predict(self, X):

        """Predict class for X.



        The predicted class of an input sample is a vote by the trees in

        the forest, weighted by their probability estimates. That is,

        the predicted class is the one with highest mean probability

        estimate across the trees.



        Parameters

        ----------

        X : array-like or sparse matrix of shape = [n_samples, n_features]

            The input samples. Internally, its dtype will be converted to

            ``dtype=np.float32``. If a sparse matrix is provided, it will be

            converted into a sparse ``csr_matrix``.



        Returns

        -------

        y : array of shape = [n_samples] or [n_samples, n_outputs]

            The predicted classes.

        """
        proba = self.predict_proba(X)
        cum_proba=proba.copy()
        for i in np.arange(1,proba.shape[1]):
            cum_proba[:,i]=cum_proba[:,i-1]+proba[:,i]


        if self.n_outputs_ == 1:

            return self.classes_.take(np.argmax(cum_proba>=0.5, axis=1), axis=0) #self.classes_.take(np.argmax(proba, axis=1), axis=0)



        else:
            raise NotImplemented
            
            
            
