import numpy as np
from boosting.adaptive import AdaptiveBoost

if __name__ == '__main__':
    feature_list = [[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]]
    label_list = [1.0, 1.0, -1.0, -1.0, 1.0]
    ada_boost = AdaptiveBoost()
    ada_boost.add_train_features(feature_list, label_list)
    ada_boost.train(9)
    print(ada_boost.classify([[5, 5], [0, 0]]))