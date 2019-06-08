## isoensemble
[![Build Status](https://travis-ci.org/chriswbartley/isoensemble.svg?branch=master)](https://travis-ci.org/chriswbartley/isoensemble)
[![Appveyor Status](https://ci.appveyor.com/api/projects/status/github/chriswbartley/isoensemble)](https://ci.appveyor.com/project/chriswbartley/isoensemble)
[![RTD Status](https://readthedocs.org/projects/isoensemble/badge/?version=latest
)](https://readthedocs.org/projects/isoensemble/badge/?version=latest)



IsoRandomForestClassifierClassifier is a Random Forest classifier with *partial* monotonicity capability (i.e. the ability to specify non-monotone features). It extends `scikit-learn's` RandomForestClassifier and inherits all `sci-kit learn` capabilities (and obviously requires `sci-kit learn`). The theory is described in Bartley C., Liu W., Reynolds M., 2017, *A Novel Framework for Partially Monotone Rule Ensembles.* ICDE submission, prepub, available [here](http://staffhome.ecm.uwa.edu.au/~19514733/). 

It is very fast and moderately accurate. 

### Code Example
First we define the monotone features, using the corresponding one-based `X` array column indices:
```
incr_feats=[6,9]
decr_feats=[1,8,13]
```
The specify the hyperparameters (see original paper for explanation):
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

### Documentation

For more examples see [the documentation](http://isoensemble.readthedocs.io/en/latest/index.html).

### Installation

To install, simply use:
```
pip install isoensemble
```

### Documentation

Documentation is provided [here](http://isoensemble.readthedocs.io/en/latest/index.html).

### Contributors

Pull requests welcome! Notes:
 - We use the
[PEP8 code formatting standard](https://www.python.org/dev/peps/pep-0008/), and
we enforce this by running a code-linter called
[`flake8`](http://flake8.pycqa.org/en/latest/) during continuous integration.
 - Continuous integration is used to run the tests in `/isoensemble/tests/test_isoensemble.py`, using [Travis](https://travis-ci.org/chriswbartley/isoensemble.svg?branch=master) (Linux) and [Appveyor](https://ci.appveyor.com/api/projects/status/github/chriswbartley/isoensemble) (Windows).
 
### License
BSD 3 Clause, Copyright (c) 2017, Christopher Bartley
All rights reserved.
