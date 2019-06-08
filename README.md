## isoensemble


IsoRandomForestClassifier is a Random Forest classifier with globally guaranteed monotonicity and partial monotonicity capability (i.e. the ability to specify both monotone and non-monotone features). It extends `scikit-learn's` RandomForestClassifier and inherits all `sci-kit learn` capabilities (and obviously requires `sci-kit learn`). It is described in Chapter 6 of the PhD thesis 'High Accuracy Partially Monotone Ordinal Classification', UWA 2019.

### Code Example
First we define the monotone features, using the corresponding one-based `X` array column indices:
```
incr_feats=[6,9]
decr_feats=[1,8,13]
```
The specify the usual RF hyperparameters:
```
# Ensure you have a reasonable number of trees
n_estimators=200
mtry = 3
```
And initialise and solve the classifier using `scikit-learn` norms:
```
clf = isoensemble.IsoRandomForestClassifier(n_estimators=n_estimators,
                                             max_features=mtry,
                                             incr_feats=incr_feats,
                                             decr_feats=decr_feats)
clf.fit(X, y)
y_pred = clf.predict(X)
```	
Of course usually the above will be embedded in some estimate of generalisation error such as out-of-box (oob) score or cross-validation.

 
### License
BSD 3 Clause, Copyright (c) 2017, Christopher Bartley
All rights reserved.
