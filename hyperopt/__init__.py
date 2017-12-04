
from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np

#hyperopt自动调参:http://hyperopt.github.io/hyperopt-sklearn/

# Download the data and split into training and test sets

digits = fetch_mldata('MNIST original')

X = digits.data
y = digits.target

test_size = int( 0.2 * len( y ) )
np.random.seed( 1 )
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

estim = HyperoptEstimator( classifier=any_classifier('clf'),
                            algo=tpe.suggest, trial_timeout=300)

estim.fit( X_train, y_train )

print( estim.score( X_test, y_test ) )
# <<show score here>>
print( estim.best_model() )
# <<show model here>>